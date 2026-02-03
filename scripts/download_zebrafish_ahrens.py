#!/usr/bin/env python3
"""
Download missing zebrafish Ahrens datasets from Janelia FigShare.

The POCO paper uses 15 zebrafish subjects (all except 8, 9, 11).
We currently have subjects 1-7, and need to download subjects 10, 12-18.

Data source: https://janelia.figshare.com/articles/dataset/Whole-brain_light-sheet_imaging_data/7272617
Reference: Chen et al., Neuron, 2018
"""

import os
import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from configs.config_global import ZEBRAFISH_AHRENS_RAW_DIR

# FigShare article info
FIGSHARE_ARTICLE_ID = "7272617"
FIGSHARE_VERSION = 4

# Subjects we need (POCO paper uses all except 8, 9, 11)
# We have: 1, 2, 3, 4, 5, 6, 7
# We need: 10, 12, 13, 14, 15, 16, 17, 18
SUBJECTS_NEEDED = [10, 12, 13, 14, 15, 16, 17, 18]

# Known file IDs from FigShare for each subject
# These are the FigShare file IDs for direct download
# Note: These may need to be updated if the dataset is modified
SUBJECT_FILE_INFO = {
    # Each subject has TimeSeries.h5 and data_full.mat files
    # The FigShare API can list the files
}


def check_existing_subjects():
    """Check which subjects we already have"""
    existing = []
    for subject_id in range(1, 19):
        subject_dir = os.path.join(ZEBRAFISH_AHRENS_RAW_DIR, f'subject_{subject_id}')
        timeseries_file = os.path.join(subject_dir, 'TimeSeries.h5')
        mat_file = os.path.join(subject_dir, 'data_full.mat')
        if os.path.exists(timeseries_file) and os.path.exists(mat_file):
            existing.append(subject_id)
    return existing


def download_subject(subject_id):
    """Download a specific subject's data from FigShare"""
    subject_dir = os.path.join(ZEBRAFISH_AHRENS_RAW_DIR, f'subject_{subject_id}')
    os.makedirs(subject_dir, exist_ok=True)

    print(f"Downloading subject {subject_id}...")

    # The FigShare dataset uses a specific folder structure
    # Files are named like: subject_N/TimeSeries.h5 and subject_N/data_full.mat

    # Use figshare API or direct download links
    # The ndownloader URL pattern for FigShare
    base_url = f"https://ndownloader.figshare.com/files/"

    # We need to get the file IDs from the FigShare API
    # For now, let's use the FigShare download tool

    # Alternative: Use FigShare CLI or requests library
    try:
        import requests

        # Get the article metadata to find file IDs
        api_url = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE_ID}/files"
        response = requests.get(api_url)

        if response.status_code == 200:
            files = response.json()
            for f in files:
                # Look for files matching this subject
                name = f['name']
                if f'subject_{subject_id}' in name or name.startswith(f'subject_{subject_id}/'):
                    download_url = f['download_url']
                    output_path = os.path.join(subject_dir, name.split('/')[-1])

                    print(f"  Downloading {name}...")
                    # Download the file
                    r = requests.get(download_url, stream=True)
                    with open(output_path, 'wb') as out_file:
                        for chunk in r.iter_content(chunk_size=8192):
                            out_file.write(chunk)
                    print(f"  Saved to {output_path}")
        else:
            print(f"  Failed to get file list: {response.status_code}")

    except ImportError:
        # Fall back to wget/curl
        print("  requests library not available, using wget...")

        # Try to use wget with the FigShare ndownloader
        # This requires knowing the specific file IDs
        pass


def download_all_missing():
    """Download all missing subjects"""
    existing = check_existing_subjects()
    print(f"Existing subjects: {existing}")

    needed = [s for s in SUBJECTS_NEEDED if s not in existing]
    print(f"Subjects to download: {needed}")

    if not needed:
        print("All required subjects already downloaded!")
        return

    # Download the full dataset archive if individual downloads fail
    print("\nNote: FigShare data may need manual download.")
    print(f"Please download from: https://janelia.figshare.com/articles/dataset/Whole-brain_light-sheet_imaging_data/{FIGSHARE_ARTICLE_ID}")
    print(f"Extract subjects {needed} to: {ZEBRAFISH_AHRENS_RAW_DIR}")

    # Try to download via the API
    for subject_id in needed:
        download_subject(subject_id)


def main():
    print("=" * 60)
    print("Zebrafish Ahrens Dataset Downloader")
    print("=" * 60)
    print(f"\nTarget directory: {ZEBRAFISH_AHRENS_RAW_DIR}")
    print(f"Required subjects: All except 8, 9, 11 (POCO paper specification)")
    print()

    # Check what we have
    existing = check_existing_subjects()
    print(f"Currently have subjects: {existing}")

    all_required = [i for i in range(1, 19) if i not in [8, 9, 11]]
    missing = [s for s in all_required if s not in existing]

    if missing:
        print(f"Missing subjects: {missing}")
        print("\n--- DOWNLOAD INSTRUCTIONS ---")
        print(f"1. Go to: https://janelia.figshare.com/articles/dataset/Whole-brain_light-sheet_imaging_data/{FIGSHARE_ARTICLE_ID}")
        print("2. Download the full dataset (~50GB)")
        print(f"3. Extract and copy the missing subject folders to: {ZEBRAFISH_AHRENS_RAW_DIR}")
        print("\nAlternatively, use figshare CLI:")
        print("  pip install figshare-cli")
        print(f"  figshare download {FIGSHARE_ARTICLE_ID}")
        print("\n")

        # Try automated download
        print("Attempting automated download...")
        download_all_missing()
    else:
        print("All required subjects present!")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
