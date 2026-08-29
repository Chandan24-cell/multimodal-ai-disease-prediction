import os
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DICOM_DIR = PROJECT_ROOT / "datasets" / "sample_dicom"


def download_sample_dicom():
    """Download sample DICOM files for testing."""
    print("Downloading sample DICOM files...")

    SAMPLE_DICOM_DIR.mkdir(parents=True, exist_ok=True)

    sample_urls = [
        "https://raw.githubusercontent.com/pydicom/pydicom/main/src/pydicom/data/test_files/CT_small.dcm",
        "https://raw.githubusercontent.com/pydicom/pydicom/main/src/pydicom/data/test_files/MR_small.dcm",
        "https://raw.githubusercontent.com/pydicom/pydicom/main/src/pydicom/data/test_files/SC_rgb_dcmtk_%2Beb%2Bcr.dcm",
    ]

    for url in sample_urls:
        try:
            filename = url.split("/")[-1]
            filepath = SAMPLE_DICOM_DIR / filename

            if not filepath.exists():
                print(f"Downloading {filename}...")
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                filepath.write_bytes(response.content)
                print(f"Downloaded: {filepath}")
            else:
                print(f"Already exists: {filepath}")

        except Exception as error:
            print(f"Failed to download {url}: {error}")

    print("\nSample DICOM files ready for testing!")
    print(f"Location: {SAMPLE_DICOM_DIR}/")


if __name__ == "__main__":
    download_sample_dicom()
