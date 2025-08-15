"""
Script to download and prepare the raw datasets used by AgentRecBench.

This helper script encapsulates the steps required to fetch the public
Amazon, Goodreads and Yelp data used by the AgentRecBench tasks.  The
official download links provided in earlier documentation have changed or
are intermittently unavailable.  This script uses the latest accessible
sources and lays out a sensible structure for the downloaded data.  Once
the downloads are complete, you can invoke the `data_process.py` script
to convert the raw data into the processed format consumed by the
simulator.

Usage:

    python download_agentrecbench_data.py --download-dir <path_to_downloads>
    # Once the raw data is in place:
    python data_process.py --input <path_to_downloads> --output <path_to_processed_data>

Notes:

    * The Amazon dataset is downloaded via the Hugging Face Hub.  Only
      the three categories used by AgentRecBench are fetched: ``Industrial_and_Scientific``,
      ``Musical_Instruments`` and ``Video_Games``.  You will need a
      Hugging Face account and may need to authenticate via the CLI or
      environment variables for large file downloads.  See
      https://huggingface.co/docs/huggingface_hub/guides/managing_repos#authentication
      for details.

    * Goodreads data is pulled directly from the UCSD domain.  These
      files are fairly large (hundreds of megabytes) and are provided as
      gzip-compressed JSON.  The script downloads and leaves them
      compressed; ``data_process.py`` can consume the ``.json.gz`` files
      directly.

    * Yelp provides an “academic dataset” containing business, user and
      review JSON files.  Downloading this dataset requires manually
      accepting Yelp’s terms of use.  You’ll be prompted to log in and
      accept the license before the data can be downloaded.  For this
      reason the script simply prints instructions rather than
      attempting an automated download.

    * If you encounter network errors when downloading large files,
      consider using a download manager such as ``aria2c``.  Replace the
      HTTP requests in the helper functions with your preferred tool.

This script is not meant to be run inside the sandbox used by the
OpenAI Code Interpreter; it is provided for users to run on their own
machines where network access is available.
"""

from __future__ import annotations

import argparse
import gzip
import os
from pathlib import Path
import shutil
import sys
from typing import Optional

import requests
from tqdm import tqdm

# Attempt to import huggingface_hub.  If it's not available the user
# should install it via pip.
try:
    from huggingface_hub import hf_hub_download
except ImportError as e:
    hf_hub_download = None


def download_with_progress(url: str, dest: Path) -> Path:
    """Download a file from a URL to a destination with a progress bar.

    Args:
        url: HTTP or HTTPS URL to download from.
        dest: Path where the file will be saved.  Parent directories
            are created if they do not exist.

    Returns:
        Path to the downloaded file.

    Raises:
        requests.HTTPError: if the HTTP request fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    progress = tqdm(total=total, unit="iB", unit_scale=True, desc=os.path.basename(dest))

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:  # filter out keep-alive chunks
                progress.update(len(chunk))
                f.write(chunk)
    progress.close()
    return dest


def download_amazon_categories(download_dir: Path) -> None:
    """Download the three Amazon categories needed for AgentRecBench via Hugging Face.

    The categories downloaded are ``Industrial_and_Scientific``,
    ``Musical_Instruments`` and ``Video_Games``.  Each category
    consists of a review file and a metadata file.  Only the review
    data is needed for AgentRecBench.

    Args:
        download_dir: Top-level directory where the files will be
            stored.  Files are placed under ``amazon`` within this
            directory.
    """
    if hf_hub_download is None:
        raise RuntimeError(
            "huggingface_hub not installed.  Please run `pip install huggingface_hub` to use this function."
        )

    categories = [
        "Industrial_and_Scientific",
        "Musical_Instruments",
        "Video_Games",
    ]
    amazon_dir = download_dir / "amazon"
    amazon_dir.mkdir(parents=True, exist_ok=True)

    for category in categories:
        print(f"Downloading Amazon category '{category}'...")
        review_filename = f"raw/review_categories/{category}.jsonl"
        # Use hf_hub_download to fetch the file.  The returned path
        # points to the cached file; we copy it into our download_dir.
        try:
            cached_path = hf_hub_download(
                repo_id="McAuley-Lab/Amazon-Reviews-2023",
                filename=review_filename,
                local_dir=amazon_dir,
                local_dir_use_symlinks=False,
            )
        except Exception as exc:
            print(
                f"Failed to download {category}.  Ensure your Hugging Face credentials are configured."
            )
            raise
        # cached_path is already saved into amazon_dir because local_dir was specified.
        print(f"Finished downloading {category} to {cached_path}")


def download_goodreads_genres(download_dir: Path) -> None:
    """Download the Goodreads genre subsets (Children, Comics & Graphic, Poetry).

    Files are saved under ``goodreads`` within the download directory.  If
    a file already exists it will not be downloaded again.

    Args:
        download_dir: Destination for the downloaded files.
    """
    goodreads_dir = download_dir / "goodreads"
    goodreads_dir.mkdir(parents=True, exist_ok=True)

    # Mapping of descriptive names to URLs and output filenames.  We
    # download only the review files; the corresponding book metadata
    # files can be fetched similarly if needed.
    urls = {
        "children_reviews": (
            "https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads/goodreads_books_children.json.gz",
            "goodreads_reviews_children.json.gz",
        ),
        "comics_graphic_reviews": (
            "https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads/goodreads_books_comics_graphic.json.gz",
            "goodreads_reviews_comics_graphic.json.gz",
        ),
        "poetry_reviews": (
            "https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads/goodreads_books_poetry.json.gz",
            "goodreads_reviews_poetry.json.gz",
        ),
    }

    for name, (url, out_name) in urls.items():
        dest = goodreads_dir / out_name
        if dest.exists():
            print(f"Skipping {name}; {dest} already exists.")
            continue
        print(f"Downloading Goodreads dataset '{name}' from {url}...")
        try:
            download_with_progress(url, dest)
        except Exception as exc:
            print(f"Failed to download {url}: {exc}")
            raise
        print(f"Finished downloading {dest}")


def download_yelp_dataset(download_dir: Path) -> None:
    """Print instructions for downloading the Yelp academic dataset.

    Yelp’s dataset requires the user to agree to the terms of service
    before downloading.  The data cannot be fetched via a simple HTTP
    request without first accepting these terms.  Accordingly this
    function prints instructions for the user rather than attempting to
    download the dataset programmatically.

    Args:
        download_dir: Directory where the user should place the Yelp
            dataset once downloaded.
    """
    yelp_dir = download_dir / "yelp"
    yelp_dir.mkdir(parents=True, exist_ok=True)
    print("""
To obtain the Yelp academic dataset, please complete the following steps manually:

  1. Visit https://www.yelp.com/dataset and click "Get the Dataset".
  2. Log in or create a Yelp account if prompted.
  3. Read and accept the terms of use for the academic dataset.
  4. Download the following files and save them into the directory:
       - yelp_academic_dataset_business.json
       - yelp_academic_dataset_user.json
       - yelp_academic_dataset_review.json
  5. Once downloaded, place them into: {yelp_dir}

These files are needed as the raw input for the data processing step.
""".format(yelp_dir=yelp_dir))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare datasets for AgentRecBench")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("./raw_agentrec_data"),
        help="Directory where raw datasets will be stored",
    )
    parser.add_argument(
        "--skip-amazon",
        action="store_true",
        help="Skip downloading Amazon categories",
    )
    parser.add_argument(
        "--skip-goodreads",
        action="store_true",
        help="Skip downloading Goodreads genre datasets",
    )
    parser.add_argument(
        "--skip-yelp",
        action="store_true",
        help="Skip printing instructions for Yelp dataset",
    )
    args = parser.parse_args()

    if not args.skip_amazon:
        try:
            download_amazon_categories(args.download_dir)
        except Exception as exc:
            print(f"Error downloading Amazon data: {exc}")
            print(
                "Ensure you have installed the huggingface_hub package and have configured credentials if required."
            )

    if not args.skip_goodreads:
        try:
            download_goodreads_genres(args.download_dir)
        except Exception as exc:
            print(f"Error downloading Goodreads data: {exc}")

    if not args.skip_yelp:
        download_yelp_dataset(args.download_dir)

    print(
        "\nAll downloads complete (or skipped).  You can now run data_process.py to convert the raw data into the processed format."
    )


if __name__ == "__main__":
    main()