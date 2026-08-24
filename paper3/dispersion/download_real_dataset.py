"""
Paper 3 — Real Experimental Dataset Downloader

Downloads a publicly accessible experimental dispersion dataset,
stores the raw file unchanged, and creates provenance metadata.

IMPORTANT:
Downloaded data must be independently identified and documented.
This script does NOT claim that every downloaded dispersion dataset
is evidence for the Lambda model.
"""

from pathlib import Path
from urllib.request import Request, urlopen
from datetime import datetime, timezone
import hashlib
import json
import sys


BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "real" / "raw"
PROVENANCE_DIR = BASE_DIR / "provenance"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

# Replace this with the actual public CSV/TSV dataset URL.
#
# Example:
# DATASET_URL = "https://example.org/dataset.csv"
#
# Do NOT use a paper landing page as if it were raw numerical data.
DATASET_URL = ""

OUTPUT_FILENAME = "experimental_dataset.csv"

SOURCE_TITLE = "Public experimental dispersion dataset"
SOURCE_AUTHORS = "To be specified"
SOURCE_DOI = "To be specified"
SOURCE_DESCRIPTION = (
    "Externally sourced experimental dispersion data for "
    "independent testing of dispersion models."
)


# ---------------------------------------------------------------------
# DOWNLOAD
# ---------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def download_file(url: str, output_path: Path):

    if not url:
        raise ValueError(
            "DATASET_URL is empty. "
            "Set it to the direct URL of a public experimental dataset."
        )

    print("=" * 70)
    print("PAPER 3 — REAL EXPERIMENTAL DATA DOWNLOADER")
    print("=" * 70)

    print()
    print("[1] SOURCE")
    print("-" * 70)
    print("URL:")
    print(url)

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Lambda-model experimental validation "
                "research pipeline"
            )
        },
    )

    print()
    print("[2] DOWNLOADING")
    print("-" * 70)

    with urlopen(request, timeout=60) as response:
        data = response.read()

    output_path.write_bytes(data)

    print(f"Downloaded bytes: {len(data):,}")
    print(f"Saved to: {output_path}")

    checksum = sha256_file(output_path)

    print()
    print("[3] CHECKSUM")
    print("-" * 70)
    print(f"SHA256: {checksum}")

    return checksum


# ---------------------------------------------------------------------
# PROVENANCE
# ---------------------------------------------------------------------

def save_provenance(checksum: str):

    provenance = {
        "dataset_type": "EXPERIMENTAL_RAW",
        "source_url": DATASET_URL,
        "source_title": SOURCE_TITLE,
        "source_authors": SOURCE_AUTHORS,
        "source_doi": SOURCE_DOI,
        "description": SOURCE_DESCRIPTION,
        "downloaded_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "filename": OUTPUT_FILENAME,
        "sha256": checksum,
        "processing": {
            "raw_file_modified": False,
            "unit_conversion": None,
            "filtering": None,
            "interpolation": None,
            "fitting": None,
        },
        "scientific_status": (
            "External experimental dataset. "
            "No claim of Lambda confirmation is made "
            "until model discrimination is performed."
        ),
    }

    path = PROVENANCE_DIR / "experimental_dataset_provenance.json"

    path.write_text(
        json.dumps(
            provenance,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("[4] PROVENANCE")
    print("-" * 70)
    print(f"Saved: {path}")


def main():

    output_path = RAW_DIR / OUTPUT_FILENAME

    try:
        checksum = download_file(
            DATASET_URL,
            output_path,
        )

        save_provenance(checksum)

    except Exception as exc:

        print()
        print("ERROR")
        print("-" * 70)
        print(str(exc))
        sys.exit(1)

    print()
    print("=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()