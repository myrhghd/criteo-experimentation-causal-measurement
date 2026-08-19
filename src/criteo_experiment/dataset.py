"""Validate and prepare the Criteo uplift dataset."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Sequence

import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from criteo_experiment.acquisition import (
    DEFAULT_RAW_PATH,
    DownloadError,
    download_dataset,
    is_valid_gzip,
)

DEFAULT_PARQUET_PATH = Path("data/processed/criteo-uplift-v2.1.parquet")
BINARY_COLUMNS = ("treatment", "exposure", "visit", "conversion")
FEATURE_COLUMNS = tuple(f"f{index}" for index in range(12))
CONTINUOUS_FEATURES = ("f0", "f2", "f7", "f10")
CATEGORICAL_FEATURES = (
    "f1",
    "f3",
    "f4",
    "f5",
    "f6",
    "f8",
    "f9",
    "f11",
)


class ValidationError(RuntimeError):
    """Indicate that dataset validation did not pass."""


def validate_feature_groups(
    continuous: Sequence[str] = CONTINUOUS_FEATURES,
    categorical: Sequence[str] = CATEGORICAL_FEATURES,
) -> None:
    """Ensure the declared feature roles partition f0 through f11."""

    continuous_set = set(continuous)
    categorical_set = set(categorical)
    overlap = continuous_set.intersection(categorical_set)
    if overlap:
        raise ValidationError(
            f"Feature groups overlap: {', '.join(sorted(overlap))}"
        )
    if len(continuous_set) != len(continuous) or len(categorical_set) != len(
        categorical
    ):
        raise ValidationError("Feature groups contain duplicate declarations")

    declared = continuous_set.union(categorical_set)
    expected = set(FEATURE_COLUMNS)
    if declared != expected:
        missing = sorted(expected.difference(declared))
        unexpected = sorted(declared.difference(expected))
        raise ValidationError(
            "Feature groups do not match f0 through f11: "
            f"missing={missing}, unexpected={unexpected}"
        )


class NumericAccumulator:
    """Maintain stable numeric summary statistics across Arrow batches."""

    def __init__(self, dtype: pa.DataType) -> None:
        self.dtype = dtype
        self.missing_count = 0
        self.nonfinite_count = 0
        self.value_count = 0
        self.minimum: int | float | None = None
        self.maximum: int | float | None = None
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, values: pa.Array) -> None:
        """Add one Arrow array to the accumulated statistics."""

        self.missing_count += values.null_count
        valid = pc.drop_null(values)
        if pa.types.is_floating(values.type):
            finite_mask = pc.is_finite(valid)
            finite_count = int(pc.sum(finite_mask).as_py() or 0)
            self.nonfinite_count += len(valid) - finite_count
            valid = pc.filter(valid, finite_mask)

        batch_count = len(valid)
        if batch_count == 0:
            return

        extrema = pc.min_max(valid).as_py()
        batch_minimum = extrema["min"]
        batch_maximum = extrema["max"]
        batch_mean = float(pc.mean(valid).as_py())
        batch_variance = float(pc.variance(valid, ddof=0).as_py() or 0.0)

        self.minimum = (
            batch_minimum
            if self.minimum is None
            else min(self.minimum, batch_minimum)
        )
        self.maximum = (
            batch_maximum
            if self.maximum is None
            else max(self.maximum, batch_maximum)
        )

        combined_count = self.value_count + batch_count
        delta = batch_mean - self.mean
        self.m2 += (
            batch_variance * batch_count
            + delta * delta * self.value_count * batch_count / combined_count
        )
        self.mean += delta * batch_count / combined_count
        self.value_count = combined_count

    def profile(self) -> dict[str, object]:
        """Return the accumulated structural profile."""

        standard_deviation = (
            math.sqrt(self.m2 / self.value_count)
            if self.value_count
            else None
        )
        return {
            "dtype": str(self.dtype),
            "missing_count": self.missing_count,
            "nonfinite_count": self.nonfinite_count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean if self.value_count else None,
            "standard_deviation": standard_deviation,
        }


def read_csv_header(path: Path) -> list[str]:
    """Read and validate the header from a gzip compressed CSV file."""

    try:
        with gzip.open(path, mode="rt", encoding="utf-8", newline="") as source:
            header = next(csv.reader(source))
    except (EOFError, OSError, StopIteration, UnicodeDecodeError) as error:
        raise ValidationError(f"Could not read the CSV header: {error}") from error

    if not header or any(not name for name in header):
        raise ValidationError("CSV header contains an empty column name")
    if len(set(header)) != len(header):
        raise ValidationError("CSV header contains duplicate column names")
    return header


def identify_columns(column_names: Sequence[str]) -> dict[str, object]:
    """Identify required experiment variables and remaining feature columns."""

    validate_feature_groups()
    missing = [name for name in BINARY_COLUMNS if name not in column_names]
    if missing:
        raise ValidationError(
            f"Dataset is missing required columns: {', '.join(missing)}"
        )

    features = [name for name in column_names if name not in BINARY_COLUMNS]
    if not features:
        raise ValidationError("Dataset does not contain any feature columns")
    return {
        "treatment": "treatment",
        "exposure": "exposure",
        "visit": "visit",
        "conversion": "conversion",
        "features": features,
        "feature_roles": {
            "continuous": list(CONTINUOUS_FEATURES),
            "categorical": list(CATEGORICAL_FEATURES),
        },
        "primary_treatment": "treatment",
    }


def _column(batch: pa.RecordBatch, name: str) -> pa.Array:
    return batch.column(batch.schema.get_field_index(name))


def _update_binary_counts(
    counts: dict[str, Counter[int]], batch: pa.RecordBatch
) -> None:
    for name in BINARY_COLUMNS:
        values = _column(batch, name)
        observed = {value for value in pc.unique(values).to_pylist() if value is not None}
        invalid = observed.difference({0, 1})
        if invalid:
            raise ValidationError(
                f"Column {name} contains invalid binary values: {sorted(invalid)}"
            )
        for value in observed:
            count = int(pc.sum(pc.equal(values, value)).as_py() or 0)
            counts[name][int(value)] += count


def _logical_violation_counts(batch: pa.RecordBatch) -> dict[str, int]:
    conversion_without_visit = int(
        pc.sum(
            pc.and_(
                pc.equal(_column(batch, "conversion"), 1),
                pc.not_equal(_column(batch, "visit"), 1),
            )
        ).as_py()
        or 0
    )
    exposure_without_treatment = int(
        pc.sum(
            pc.and_(
                pc.equal(_column(batch, "exposure"), 1),
                pc.not_equal(_column(batch, "treatment"), 1),
            )
        ).as_py()
        or 0
    )
    return {
        "conversion_without_visit": conversion_without_visit,
        "exposure_without_treatment": exposure_without_treatment,
    }


def _output_schema(raw_schema: pa.Schema) -> pa.Schema:
    fields = [
        pa.field(field.name, pa.int8(), nullable=field.nullable)
        if field.name in BINARY_COLUMNS
        else field
        for field in raw_schema
    ]
    return pa.schema(fields)


def _cast_batch(batch: pa.RecordBatch, schema: pa.Schema) -> pa.RecordBatch:
    arrays = []
    for field in schema:
        values = _column(batch, field.name)
        arrays.append(
            pc.cast(values, field.type, safe=True)
            if values.type != field.type
            else values
        )
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def _binary_profiles(
    counts: dict[str, Counter[int]],
    statistics: dict[str, NumericAccumulator],
) -> dict[str, dict[str, object]]:
    profiles = {}
    for name in BINARY_COLUMNS:
        nonmissing_count = sum(counts[name].values())
        value_counts = {
            str(value): counts[name].get(value, 0) for value in (0, 1)
        }
        profiles[name] = {
            "dtype": str(statistics[name].dtype),
            "missing_count": statistics[name].missing_count,
            "observed_values": [
                value for value in (0, 1) if counts[name].get(value, 0)
            ],
            "counts": value_counts,
            "proportions": {
                value: count / nonmissing_count if nonmissing_count else None
                for value, count in value_counts.items()
            },
        }
    return profiles


def _approximate_cardinalities(
    parquet_path: Path, feature_columns: Sequence[str]
) -> dict[str, int]:
    expressions = [
        pl.col(name).approx_n_unique().alias(name) for name in feature_columns
    ]
    result = pl.scan_parquet(parquet_path).select(expressions).collect(
        engine="streaming"
    )
    return {name: int(result[name][0]) for name in feature_columns}


def _count_duplicate_rows(parquet_path: Path, total_rows: int) -> int:
    unique_rows = (
        pl.scan_parquet(parquet_path)
        .unique(maintain_order=False)
        .select(pl.len().alias("unique_rows"))
        .collect(engine="streaming")["unique_rows"][0]
    )
    return total_rows - int(unique_rows)


def _summarize_parquet(
    parquet_path: Path,
    feature_columns: Sequence[str],
) -> dict[str, object]:
    parquet_file = pq.ParquetFile(parquet_path)
    schema = parquet_file.schema_arrow
    statistics = {
        field.name: NumericAccumulator(field.type) for field in schema
    }
    binary_counts = {name: Counter() for name in BINARY_COLUMNS}
    logical_violations = {
        "conversion_without_visit": 0,
        "exposure_without_treatment": 0,
    }
    row_count = 0

    for batch in parquet_file.iter_batches(batch_size=262_144):
        row_count += batch.num_rows
        for name, accumulator in statistics.items():
            accumulator.add(_column(batch, name))
        _update_binary_counts(binary_counts, batch)
        violations = _logical_violation_counts(batch)
        for name, count in violations.items():
            logical_violations[name] += count

    profiles = {name: accumulator.profile() for name, accumulator in statistics.items()}
    return {
        "path": str(parquet_path),
        "size_bytes": parquet_path.stat().st_size,
        "row_count": row_count,
        "column_count": len(schema.names),
        "column_names": schema.names,
        "schema": [
            {"name": field.name, "dtype": str(field.type)} for field in schema
        ],
        "missing_counts": {
            name: profile["missing_count"] for name, profile in profiles.items()
        },
        "nonfinite_counts": {
            name: profile["nonfinite_count"] for name, profile in profiles.items()
        },
        "binary_columns": _binary_profiles(binary_counts, statistics),
        "feature_profiles": {name: profiles[name] for name in feature_columns},
        "logical_violations": logical_violations,
    }


def _statistics_match(
    left: dict[str, object], right: dict[str, object]
) -> bool:
    for key in ("minimum", "maximum", "mean", "standard_deviation"):
        left_value = left[key]
        right_value = right[key]
        if left_value is None or right_value is None:
            if left_value != right_value:
                return False
        elif not math.isclose(
            float(left_value),
            float(right_value),
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            return False
    return True


def _validate_equivalence(
    raw_report: dict[str, object],
    parquet_report: dict[str, object],
    feature_columns: Sequence[str],
) -> dict[str, object]:
    checks = {
        "row_count": raw_report["row_count"] == parquet_report["row_count"],
        "column_names": raw_report["column_names"]
        == parquet_report["column_names"],
        "missing_counts": raw_report["missing_counts"]
        == parquet_report["missing_counts"],
        "binary_counts": all(
            raw_report["binary_columns"][name]["counts"]
            == parquet_report["binary_columns"][name]["counts"]
            for name in BINARY_COLUMNS
        ),
        "feature_statistics": all(
            _statistics_match(
                raw_report["feature_profiles"][name],
                parquet_report["feature_profiles"][name],
            )
            for name in feature_columns
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
    }


def prepare_dataset(
    raw_path: Path = DEFAULT_RAW_PATH,
    parquet_path: Path = DEFAULT_PARQUET_PATH,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Validate a raw gzip CSV and create an equivalent Parquet dataset."""

    raw_path = Path(raw_path)
    parquet_path = Path(parquet_path)
    if not raw_path.is_file():
        raise ValidationError(f"Raw dataset does not exist: {raw_path}")
    if not is_valid_gzip(raw_path):
        raise ValidationError(f"Raw dataset is not a complete gzip stream: {raw_path}")
    if parquet_path.exists() and not overwrite:
        raise ValidationError(
            f"Processed dataset already exists: {parquet_path}. Use --overwrite to replace it."
        )

    header = read_csv_header(raw_path)
    roles = identify_columns(header)
    feature_columns = roles["features"]
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = parquet_path.with_name(f"{parquet_path.name}.part")
    if temporary_path.exists():
        raise ValidationError(
            f"Temporary processed file already exists: {temporary_path}"
        )

    raw_report: dict[str, object]
    writer: pq.ParquetWriter | None = None
    try:
        with pa.input_stream(str(raw_path), compression="gzip") as source:
            reader = pacsv.open_csv(
                source,
                read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024),
                parse_options=pacsv.ParseOptions(delimiter=","),
            )
            raw_schema = reader.schema
            if raw_schema.names != header:
                raise ValidationError(
                    "Arrow schema columns do not match the observed CSV header"
                )
            nonnumeric = [
                field.name
                for field in raw_schema
                if not (
                    pa.types.is_integer(field.type)
                    or pa.types.is_floating(field.type)
                )
            ]
            if nonnumeric:
                raise ValidationError(
                    f"Dataset contains nonnumeric columns: {', '.join(nonnumeric)}"
                )

            statistics = {
                field.name: NumericAccumulator(field.type) for field in raw_schema
            }
            binary_counts = {name: Counter() for name in BINARY_COLUMNS}
            logical_violations = {
                "conversion_without_visit": 0,
                "exposure_without_treatment": 0,
            }
            row_count = 0
            processed_schema = _output_schema(raw_schema)
            writer = pq.ParquetWriter(
                temporary_path,
                processed_schema,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )

            for batch in reader:
                row_count += batch.num_rows
                for name, accumulator in statistics.items():
                    accumulator.add(_column(batch, name))
                _update_binary_counts(binary_counts, batch)
                violations = _logical_violation_counts(batch)
                for name, count in violations.items():
                    logical_violations[name] += count
                writer.write_batch(_cast_batch(batch, processed_schema))

            writer.close()
            writer = None

        if row_count == 0:
            raise ValidationError("Dataset contains no data rows")

        profiles = {
            name: accumulator.profile() for name, accumulator in statistics.items()
        }
        nonfinite_counts = {
            name: profile["nonfinite_count"] for name, profile in profiles.items()
        }
        if any(nonfinite_counts.values()):
            raise ValidationError(
                f"Dataset contains nonfinite numeric values: {nonfinite_counts}"
            )

        cardinalities = _approximate_cardinalities(
            temporary_path, feature_columns
        )
        feature_profiles = {name: profiles[name] for name in feature_columns}
        for name, cardinality in cardinalities.items():
            feature_profiles[name]["approximate_cardinality"] = cardinality

        raw_report = {
            "path": str(raw_path),
            "size_bytes": raw_path.stat().st_size,
            "gzip_integrity": True,
            "header_readable": True,
            "consistent_column_count": True,
            "row_count": row_count,
            "column_count": len(header),
            "column_names": header,
            "schema": [
                {"name": field.name, "dtype": str(field.type)}
                for field in raw_schema
            ],
            "identified_columns": roles,
            "missing_counts": {
                name: profile["missing_count"] for name, profile in profiles.items()
            },
            "nonfinite_counts": nonfinite_counts,
            "binary_columns": _binary_profiles(binary_counts, statistics),
            "feature_profiles": feature_profiles,
            "duplicate_full_rows": _count_duplicate_rows(
                temporary_path, row_count
            ),
            "logical_violations": logical_violations,
        }

        parquet_report = _summarize_parquet(temporary_path, feature_columns)
        equivalence = _validate_equivalence(
            raw_report, parquet_report, feature_columns
        )
        if not equivalence["passed"]:
            raise ValidationError(
                f"Parquet equivalence checks failed: {equivalence['checks']}"
            )

        os.replace(temporary_path, parquet_path)
        parquet_report["path"] = str(parquet_path)
        return {
            "raw": raw_report,
            "parquet": parquet_report,
            "equivalence": equivalence,
        }
    except Exception as error:
        if writer is not None:
            writer.close()
        if temporary_path.exists():
            temporary_path.unlink()
        if isinstance(error, ValidationError):
            raise
        raise ValidationError(f"Dataset preparation failed: {error}") from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Acquire and prepare the official Criteo uplift dataset."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    acquire = commands.add_parser("acquire", help="Download the raw dataset.")
    acquire.add_argument("--destination", type=Path, default=DEFAULT_RAW_PATH)

    prepare = commands.add_parser(
        "prepare", help="Validate raw data and create Parquet."
    )
    prepare.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    prepare.add_argument("--output", type=Path, default=DEFAULT_PARQUET_PATH)
    prepare.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """Run the dataset acquisition and preparation command line interface."""

    parser = _build_parser()
    arguments = parser.parse_args()
    try:
        if arguments.command == "acquire":
            report = download_dataset(arguments.destination).to_dict()
        else:
            report = prepare_dataset(
                arguments.raw,
                arguments.output,
                overwrite=arguments.overwrite,
            )
    except (DownloadError, ValidationError) as error:
        parser.exit(1, f"error: {error}\n")

    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
