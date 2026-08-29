import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from inference.dicom_loader import DICOMImageLoader


def test_dicom_loader():
    """Test DICOM loading with sample files."""
    dicom_dir = PROJECT_ROOT / "datasets" / "sample_dicom"

    if not dicom_dir.exists():
        print("DICOM directory not found. Run: python scripts/download_sample_dicom.py")
        return

    dicom_files = list(dicom_dir.glob("*.dcm"))

    if not dicom_files:
        print("No DICOM files found in datasets/sample_dicom/")
        return

    print(f"Found {len(dicom_files)} DICOM file(s)\n")

    for dicom_file in dicom_files:
        print(f"\n{'=' * 60}")
        print(f"Testing: {dicom_file.name}")
        print("=" * 60)

        try:
            result = DICOMImageLoader.load_dicom_file(dicom_file)

            print("Successfully loaded!")
            print("\nMetadata:")
            for key, value in result["metadata"].items():
                if value is not None:
                    print(f"  {key}: {value}")

            print("\nImage:")
            print(f"  Size: {result['image'].size}")
            print(f"  Mode: {result['image'].mode}")

            output_path = dicom_dir / f"{dicom_file.stem}.png"
            result["image"].save(output_path)
            print(f"\nSaved converted image: {output_path}")

        except Exception as error:
            print(f"Error: {error}")
            raise

    print(f"\n{'=' * 60}")
    print("DICOM loader test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_dicom_loader()
