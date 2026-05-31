"""
Script to download test images from the NIH ChestX-ray14 dataset.

Downloads representative images for the 5 model pathologies:
  - Cardiomegaly
  - Edema
  - Consolidation
  - Atelectasis
  - Pleural Effusion
  - No Finding (healthy images, for negative control testing)

Usage:
  uv run python scripts/download_test_images.py

The NIH ChestX-ray14 dataset is available on Kaggle:
  https://www.kaggle.com/datasets/nih-chest-xrays/data

Prerequisites:
  1. Create a Kaggle account (free) at https://www.kaggle.com
  2. Go to Settings → API → Generate New Token → download kaggle.json
  3. Add your credentials to the .env file in the project root:
       KAGGLE_USERNAME=your_username
       KAGGLE_KEY=your_api_key
  4. Run: uv sync --extra dev
  5. Run: uv run python scripts/download_test_images.py
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET = "nih-chest-xrays/data"
CSV_FILE = "Data_Entry_2017.csv"

TARGET_CLASSES = [
    "Cardiomegaly",
    "Edema",
    "Consolidation",
    "Atelectasis",
    "Pleural Effusion",
    "No Finding",  # healthy chest X-rays (negative control)
]

# NIH ChestX-ray14 uses different label names than CheXpert.
# This mapping translates model class names → NIH CSV label names.
NIH_LABEL_MAP = {
    "Cardiomegaly": "Cardiomegaly",
    "Edema": "Edema",
    "Consolidation": "Consolidation",
    "Atelectasis": "Atelectasis",
    "Pleural Effusion": "Effusion",   # NIH uses "Effusion", not "Pleural Effusion"
    "No Finding": "No Finding",
}

OUTPUT_DIR = Path("tests/images")

# Images are split across 12 zip archives in the dataset.
# Each archive covers a range of image files.
IMAGE_ARCHIVES = [
    "images_001", "images_002", "images_003", "images_004",
    "images_005", "images_006", "images_007", "images_008",
    "images_009", "images_010", "images_011", "images_012",
]

# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def load_env() -> dict[str, str]:
    """Load .env values from the project root, merged with environment variables."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    env_values = dict(dotenv_values(env_file)) if env_file.exists() else {}
    # Environment variables take precedence
    env_values.update({k: v for k, v in os.environ.items() if k.startswith("KAGGLE_")})
    return env_values


def validate_credentials(env: dict[str, str]) -> tuple[str, str]:
    """Return (username, key) or raise ValueError if not configured."""
    username = env.get("KAGGLE_USERNAME", "")
    key = env.get("KAGGLE_KEY", "")
    if not username or not key or username == "your_kaggle_username":
        raise ValueError(
            "Kaggle credentials not configured.\n"
            "Add to your .env file:\n"
            "  KAGGLE_USERNAME=your_username\n"
            "  KAGGLE_KEY=your_api_key\n\n"
            "Get your token at: https://www.kaggle.com/settings → API → Generate New Token"
        )
    return username, key


def get_images_per_class(env: dict[str, str]) -> int:
    """Read KAGGLE_IMAGES_PER_CLASS from env, default 10."""
    try:
        return max(1, int(env.get("KAGGLE_IMAGES_PER_CLASS", "10")))
    except (ValueError, TypeError):
        return 10


# ---------------------------------------------------------------------------
# Kaggle CLI helpers
# ---------------------------------------------------------------------------


def kaggle_cmd(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    """Run a kaggle CLI command with credentials injected via environment."""
    proc_env = os.environ.copy()
    proc_env["KAGGLE_USERNAME"] = env.get("KAGGLE_USERNAME", "")
    proc_env["KAGGLE_KEY"] = env.get("KAGGLE_KEY", "")

    return subprocess.run(
        [sys.executable, "-m", "kaggle"] + args,
        env=proc_env,
        capture_output=True,
        text=True,
    )


def download_file(filename: str, dest_dir: Path, env: dict[str, str]) -> bool:
    """Download a single file from the dataset using the kaggle CLI."""
    result = kaggle_cmd(
        ["datasets", "download", DATASET, "-f", filename, "-p", str(dest_dir), "--unzip"],
        env,
    )
    if result.returncode != 0:
        print(f"    ✗ kaggle error: {result.stderr.strip()[:200]}")
        return False
    return True


# ---------------------------------------------------------------------------
# CSV and image selection
# ---------------------------------------------------------------------------


def download_csv(dest_dir: Path, env: dict[str, str]) -> pd.DataFrame:
    """Download and parse the labels CSV."""
    import zipfile

    print(f"Downloading {CSV_FILE}")
    csv_path = dest_dir / CSV_FILE
    zip_path = dest_dir / f"{CSV_FILE}.zip"

    # Already extracted — use directly
    if csv_path.exists():
        return pd.read_csv(csv_path)

    # Download if zip not present yet
    if not zip_path.exists():
        ok = download_file(CSV_FILE, dest_dir, env)
        if not ok:
            raise RuntimeError(f"Failed to download {CSV_FILE}")

    # Unzip manually (kaggle CLI on Windows doesn't always honour --unzip)
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest_dir)
        zip_path.unlink()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path} after download. "
            f"Files present: {[f.name for f in dest_dir.iterdir()]}"
        )

    return pd.read_csv(csv_path)


def select_images(df: pd.DataFrame, images_per_class: int) -> dict[str, list[str]]:
    """Select images_per_class filenames for each target class.

    Uses NIH_LABEL_MAP to translate model class names to NIH CSV label names.
    """
    selected: dict[str, list[str]] = {}

    for cls in TARGET_CLASSES:
        nih_label = NIH_LABEL_MAP[cls]

        if nih_label == "No Finding":
            mask = df["Finding Labels"] == "No Finding"
        else:
            # Column may contain multiple labels separated by |
            mask = df["Finding Labels"].str.contains(nih_label, regex=False)

        candidates = df[mask]["Image Index"].tolist()
        selected[cls] = candidates[:images_per_class]
        print(
            f"  {cls} (NIH: '{nih_label}'): {len(selected[cls])} images selected "
            f"(out of {len(candidates)} available)"
        )

    return selected


def find_archive_for(filename: str, df: pd.DataFrame) -> str | None:
    """Find which archive zip contains the given image filename.

    The dataset splits images across images_001 through images_012.
    We use the numeric prefix of the filename to estimate the archive.
    Each archive contains roughly 9,000 images.
    """
    try:
        # Filenames are like 00000001_000.png — use the patient number
        patient_id = int(filename.split("_")[0])
        # Each archive covers ~9000 patients; archive index = patient_id // 9000
        archive_idx = min(patient_id // 9000, len(IMAGE_ARCHIVES) - 1)
        return IMAGE_ARCHIVES[archive_idx]
    except (ValueError, IndexError):
        return None


def download_image(filename: str, dest: Path, env: dict[str, str], df: pd.DataFrame) -> bool:
    """Download a single image from the dataset."""
    import zipfile

    archive = find_archive_for(filename, df)
    if archive is None:
        print(f"    ✗ Could not determine archive for {filename}")
        return False

    # The file path inside the dataset is: images_00X/images/<filename>
    file_path = f"{archive}/images/{filename}"

    result = kaggle_cmd(
        ["datasets", "download", DATASET, "-f", file_path, "-p", str(dest.parent), "--unzip"],
        env,
    )

    if result.returncode != 0:
        print(f"    ✗ kaggle error: {result.stderr.strip()[:200]}")
        return False

    # Unzip manually if CLI left a .zip (common on Windows)
    zip_path = dest.parent / f"{filename}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(dest.parent)
        zip_path.unlink()

    return dest.exists() and dest.stat().st_size > 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("Test image download — NIH ChestX-ray14")
    print("=" * 60)

    # Load credentials and settings
    try:
        env = load_env()
        username, _ = validate_credentials(env)
        images_per_class = get_images_per_class(env)
        print(f"✓ Credentials loaded ({username}, images per class: {images_per_class})")
    except ValueError as e:
        print(f"\n✗ {e}")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download labels CSV
    try:
        df = download_csv(OUTPUT_DIR, env)
        print(f"✓ CSV loaded: {len(df):,} images in the dataset")
    except Exception as e:
        print(f"\n✗ Failed to download CSV: {e}")
        return

    # Select images per class
    print("\nSelecting images by pathology:")
    selected = select_images(df, images_per_class)

    # Download images
    print(f"\nDownloading images to '{OUTPUT_DIR}/'...")
    total_ok = 0
    total_fail = 0

    for cls, filenames in selected.items():
        cls_dir = OUTPUT_DIR / cls.replace(" ", "_")
        cls_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{cls}]")
        for filename in filenames:
            dest = cls_dir / filename
            if dest.exists() and dest.stat().st_size > 0:
                print(f"    ✓ {filename} (already exists)")
                total_ok += 1
                continue

            ok = download_image(filename, dest, env, df)
            if ok:
                print(f"    ✓ {filename}")
                total_ok += 1
            else:
                total_fail += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Download complete: {total_ok} OK, {total_fail} failed")
    print(f"Images saved to: {OUTPUT_DIR.resolve()}")
    print("\nDirectory structure:")
    for cls in TARGET_CLASSES:
        cls_dir = OUTPUT_DIR / cls.replace(" ", "_")
        if cls_dir.exists():
            count = len(list(cls_dir.glob("*.png")))
            print(f"  {cls_dir}/ ({count} images)")
    print("=" * 60)
    print("\nTo test the endpoint:")
    print("  curl -X POST http://localhost:8000/predict \\")
    print(f'    -F "file=@{OUTPUT_DIR}/Cardiomegaly/<filename>.png"')


if __name__ == "__main__":
    main()
