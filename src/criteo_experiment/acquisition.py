"""Download the official Criteo uplift dataset safely."""

from __future__ import annotations

import gzip
import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

OFFICIAL_DATASET_URL = (
    "http://go.criteo.net/criteo-research-uplift-v2.1.csv.gz"
)
DATASET_FILENAME = "criteo-research-uplift-v2.1.csv.gz"
EXPECTED_DATASET_SHA256 = (
    "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
)
DEFAULT_RAW_PATH = Path("data/raw") / DATASET_FILENAME
ALLOWED_DOWNLOAD_HOSTS = {
    "go.criteo.net",
    "criteostorage.blob.core.windows.net",
}


class DownloadError(RuntimeError):
    """Indicate that dataset acquisition did not complete safely."""


@dataclass(frozen=True)
class DownloadRecord:
    """Describe a verified local dataset download."""

    path: str
    source_url: str
    resolved_url: str | None
    size_bytes: int
    sha256: str
    reused_existing_file: bool

    def to_dict(self) -> dict[str, object]:
        """Return a representation suitable for JSON output."""

        return asdict(self)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Calculate the SHA256 digest of a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid_gzip(path: Path, chunk_size: int = 8 * 1024 * 1024) -> bool:
    """Return whether a nonempty file is a complete gzip stream."""

    if not path.is_file() or path.stat().st_size == 0:
        return False

    decompressed_bytes = 0
    try:
        with gzip.open(path, "rb") as source:
            while chunk := source.read(chunk_size):
                decompressed_bytes += len(chunk)
    except (EOFError, OSError):
        return False
    return decompressed_bytes > 0


def download_dataset(
    destination: Path = DEFAULT_RAW_PATH,
    *,
    url: str = OFFICIAL_DATASET_URL,
    expected_sha256: str | None = EXPECTED_DATASET_SHA256,
    chunk_size: int = 8 * 1024 * 1024,
    timeout_seconds: int = 60,
) -> DownloadRecord:
    """Stream the official dataset to a verified local gzip file.

    An existing complete gzip file is reused. An invalid destination is never
    overwritten automatically.
    """

    destination = Path(destination)
    if destination.exists():
        if not is_valid_gzip(destination, chunk_size):
            raise DownloadError(
                f"Existing file is not a complete gzip stream: {destination}"
            )
        actual_sha256 = sha256_file(destination, chunk_size)
        _verify_sha256(actual_sha256, expected_sha256)
        return DownloadRecord(
            path=str(destination),
            source_url=url,
            resolved_url=None,
            size_bytes=destination.stat().st_size,
            sha256=actual_sha256,
            reused_existing_file=True,
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f"{destination.name}.part")
    if temporary_path.exists():
        raise DownloadError(
            f"Partial download already exists and requires review: {temporary_path}"
        )

    request = Request(
        url,
        headers={"User-Agent": "criteo-experimentation/0.1"},
    )
    bytes_written = 0
    digest = hashlib.sha256()
    resolved_url = url
    created_temporary_file = False

    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            status = response.getcode()
            if status != 200:
                raise DownloadError(f"Dataset request returned HTTP {status}")

            resolved_url = response.geturl()
            resolved = urlparse(resolved_url)
            if url == OFFICIAL_DATASET_URL and (
                resolved.scheme != "https"
                or resolved.hostname not in ALLOWED_DOWNLOAD_HOSTS
            ):
                raise DownloadError(
                    f"Official dataset redirected to an unexpected URL: {resolved_url}"
                )

            content_length_value = response.headers.get("Content-Length")
            if not content_length_value:
                raise DownloadError("Dataset response did not provide Content-Length")
            expected_bytes = int(content_length_value)
            if expected_bytes <= 0:
                raise DownloadError("Dataset response reported an empty file")

            with temporary_path.open("xb") as target:
                created_temporary_file = True
                while chunk := response.read(chunk_size):
                    target.write(chunk)
                    digest.update(chunk)
                    bytes_written += len(chunk)

        if bytes_written != expected_bytes:
            raise DownloadError(
                "Dataset download was incomplete: "
                f"expected {expected_bytes} bytes, received {bytes_written}"
            )
        if not is_valid_gzip(temporary_path, chunk_size):
            raise DownloadError("Downloaded file is not a complete gzip stream")
        _verify_sha256(digest.hexdigest(), expected_sha256)

        os.replace(temporary_path, destination)
    except Exception as error:
        if created_temporary_file and temporary_path.exists():
            temporary_path.unlink()
        if isinstance(error, DownloadError):
            raise
        raise DownloadError(f"Dataset download failed: {error}") from error

    return DownloadRecord(
        path=str(destination),
        source_url=url,
        resolved_url=resolved_url,
        size_bytes=bytes_written,
        sha256=digest.hexdigest(),
        reused_existing_file=False,
    )


def _verify_sha256(actual: str, expected: str | None) -> None:
    if expected is not None and actual.lower() != expected.lower():
        raise DownloadError(
            "Dataset SHA256 mismatch: "
            f"expected {expected.lower()}, received {actual.lower()}"
        )
