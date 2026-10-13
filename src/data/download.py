"""Download dataset archives for Math2LaTeX.

Fetches MathWriting dataset (excerpt or full) and extracts to data/raw/.
Provides clear progress bars and error messages.
"""

import argparse
import os
from pathlib import Path
import sys
import tarfile
import urllib.error
import urllib.request
from tqdm import tqdm


DATASET_URLS = {
    "excerpt": "https://storage.googleapis.com/mathwriting_data/mathwriting-2024-excerpt.tgz",
    "full": "https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz",
}


class DownloadProgressBar(tqdm):
    """Visual progress bar for urllib download hook."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file from a URL to dest_path with progress bar.
    
    Args:
        url: Remote URL to download.
        dest_path: Destination file path on local disk.
        
    Returns:
        True if successful, False otherwise.
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from: {url}")
    print(f"Saving to: {dest_path}")
    
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=dest_path.name) as t:
            urllib.request.urlretrieve(url, filename=str(dest_path), reporthook=t.update_to)
        print("Download completed successfully.")
        return True
    except urllib.error.URLError as e:
        print(f"\n[ERROR] Failed to download {url}: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during download: {e}")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """Extract .tgz / .tar.gz archive.
    
    Args:
        archive_path: Path to archive file.
        extract_to: Destination directory.
        
    Returns:
        True if extracted successfully, False otherwise.
    """
    print(f"Extracting {archive_path.name} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    try:
        if str(archive_path).endswith((".tgz", ".tar.gz")):
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=extract_to)
        elif str(archive_path).endswith(".tar"):
            with tarfile.open(archive_path, "r:") as tar:
                tar.extractall(path=extract_to)
        else:
            print(f"[WARNING] Unsupported archive extension for {archive_path}")
            return False
        print("Extraction complete.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to extract archive: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download math handwriting datasets")
    parser.add_argument(
        "--dataset",
        choices=["excerpt", "full"],
        default="excerpt",
        help="Dataset choice: 'excerpt' (~1.5 MB for quick verification) or 'full' (~2.9 GB)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/raw",
        help="Target folder for raw data archives",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    url = DATASET_URLS[args.dataset]
    archive_name = url.split("/")[-1]
    archive_path = out_dir / archive_name
    extracted_dir = out_dir / "mathwriting"

    # Only download if archive does not already exist
    if not archive_path.exists():
        success = download_file(url, archive_path)
        if not success:
            sys.exit(1)
    else:
        print(f"Archive already exists at {archive_path}")

    # Extract archive
    extract_archive(archive_path, extracted_dir)
    print(f"MathWriting data ready at: {extracted_dir.resolve()}")


if __name__ == "__main__":
    main()
