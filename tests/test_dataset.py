from __future__ import annotations

import csv
import gzip
import io
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import criteo_experiment.acquisition as acquisition
from criteo_experiment.acquisition import (
    DownloadError,
    download_dataset,
    sha256_file,
)
from criteo_experiment.dataset import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    FEATURE_COLUMNS,
    ValidationError,
    identify_columns,
    prepare_dataset,
    validate_feature_groups,
)

FIXTURE_COLUMNS = [
    "f0",
    "f1",
    "treatment",
    "conversion",
    "visit",
    "exposure",
]


def _write_fixture(
    path: Path,
    rows: list[dict[str, int | float]],
    columns: list[str] = FIXTURE_COLUMNS,
) -> None:
    with gzip.open(path, mode="wt", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _valid_rows() -> list[dict[str, int | float]]:
    treated_conversion = {
        "f0": 1.5,
        "f1": 10.0,
        "treatment": 1,
        "conversion": 1,
        "visit": 1,
        "exposure": 1,
    }
    return [
        treated_conversion,
        {
            "f0": 2.5,
            "f1": 20.0,
            "treatment": 0,
            "conversion": 0,
            "visit": 0,
            "exposure": 0,
        },
        treated_conversion.copy(),
    ]


def test_identify_columns_rejects_missing_required_column() -> None:
    columns = [name for name in FIXTURE_COLUMNS if name != "exposure"]

    with pytest.raises(ValidationError, match="exposure"):
        identify_columns(columns)


def test_feature_groups_partition_all_declared_features() -> None:
    validate_feature_groups()

    declared = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
    assert len(declared) == len(set(declared))
    assert set(declared) == set(FEATURE_COLUMNS)


def test_feature_groups_reject_overlap() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        validate_feature_groups(
            continuous=("f0",),
            categorical=FEATURE_COLUMNS,
        )


def test_prepare_dataset_rejects_invalid_binary_value(tmp_path: Path) -> None:
    raw_path = tmp_path / "invalid.csv.gz"
    parquet_path = tmp_path / "invalid.parquet"
    rows = _valid_rows()
    rows[0]["treatment"] = 2
    _write_fixture(raw_path, rows)

    with pytest.raises(ValidationError, match="invalid binary values"):
        prepare_dataset(raw_path, parquet_path)

    assert not parquet_path.exists()


def test_prepare_dataset_preserves_rows_and_validates_equivalence(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "fixture.csv.gz"
    parquet_path = tmp_path / "fixture.parquet"
    _write_fixture(raw_path, _valid_rows())

    report = prepare_dataset(raw_path, parquet_path)

    assert report["raw"]["row_count"] == 3
    assert report["raw"]["duplicate_full_rows"] == 1
    assert report["equivalence"]["passed"] is True
    assert pq.ParquetFile(parquet_path).metadata.num_rows == 3
    assert pq.read_schema(parquet_path).field("treatment").type.bit_width == 8
    parquet_rows = pq.read_table(parquet_path).to_pylist()
    assert parquet_rows[0] == parquet_rows[2]


def test_download_reuses_valid_existing_gzip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "existing.csv.gz"
    _write_fixture(destination, _valid_rows())

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network should not be used for an existing valid file")

    monkeypatch.setattr(acquisition, "urlopen", fail_if_called)

    expected_sha256 = sha256_file(destination)
    record = download_dataset(destination, expected_sha256=expected_sha256)

    assert record.reused_existing_file is True
    assert record.sha256 == expected_sha256
    assert record.size_bytes == destination.stat().st_size


def test_existing_gzip_with_checksum_mismatch_is_not_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "existing.csv.gz"
    _write_fixture(destination, _valid_rows())
    original_bytes = destination.read_bytes()

    def fail_if_called(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network should not be used for an existing file")

    monkeypatch.setattr(acquisition, "urlopen", fail_if_called)

    with pytest.raises(DownloadError, match="SHA256 mismatch"):
        download_dataset(destination, expected_sha256="0" * 64)

    assert destination.read_bytes() == original_bytes


class _IncompleteResponse(io.BytesIO):
    headers = {"Content-Length": "10"}

    def getcode(self) -> int:
        return 200

    def geturl(self) -> str:
        return (
            "https://criteostorage.blob.core.windows.net/"
            "criteo-research-datasets/criteo-uplift-v2.1.csv.gz"
        )

    def __enter__(self) -> _IncompleteResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def test_download_rejects_incomplete_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "incomplete.csv.gz"
    monkeypatch.setattr(
        acquisition,
        "urlopen",
        lambda *args, **kwargs: _IncompleteResponse(b"abc"),
    )

    with pytest.raises(DownloadError, match="incomplete"):
        download_dataset(destination, chunk_size=2)

    assert not destination.exists()
    assert not destination.with_name(f"{destination.name}.part").exists()


def test_download_rejects_completed_checksum_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "mismatch.csv.gz"
    payload_path = tmp_path / "payload.csv.gz"
    _write_fixture(payload_path, _valid_rows())
    payload = payload_path.read_bytes()
    response = _IncompleteResponse(payload)
    response.headers = {"Content-Length": str(len(payload))}
    monkeypatch.setattr(acquisition, "urlopen", lambda *args, **kwargs: response)

    with pytest.raises(DownloadError, match="SHA256 mismatch"):
        download_dataset(destination, expected_sha256="0" * 64)

    assert not destination.exists()
    assert not destination.with_name(f"{destination.name}.part").exists()
