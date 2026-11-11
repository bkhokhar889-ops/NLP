"""Task 1 - download / load the raw SMS Spam Collection.

The raw data is NOT committed to Git. Run this script (or src.run_pipeline, which calls it)
to fetch it into data/raw/.

    python -m src.load_data                 # download if missing, then print a summary
    python -m src.load_data --path my.csv   # load a manually downloaded file instead
"""
import argparse
import csv
import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from src import config


def download_raw(force: bool = False) -> Path:
    """Download and unzip the UCI archive into data/raw/. Returns the path of the raw file."""
    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    if config.RAW_FILE.exists() and not force:
        return config.RAW_FILE
    print(f"Downloading {config.DATASET_URL} ...")
    try:
        with urllib.request.urlopen(config.DATASET_URL, timeout=60) as resp:
            payload = resp.read()
    except Exception as exc:  # network problems, proxies, etc.
        raise SystemExit(
            f"\nCould not download the dataset ({exc}).\n"
            f"Download it manually from {config.DATASET_PAGE}\n"
            f"unzip it and place the file called 'SMSSpamCollection' in {config.DATA_RAW}/\n"
            "(or pass your own file with --path)."
        )
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        config.RAW_FILE.write_bytes(zf.read("SMSSpamCollection"))
    print(f"Saved raw data to {config.RAW_FILE}")
    return config.RAW_FILE


def load_raw(path=None) -> pd.DataFrame:
    """Load the raw data into a DataFrame with columns: label, text, label_num.

    Supports the original UCI file (tab separated, no header) and the common Kaggle
    'spam.csv' variant (columns v1, v2, latin-1 encoded).
    """
    path = Path(path) if path else config.RAW_FILE
    if not path.exists():
        path = download_raw()

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, encoding="latin-1", usecols=[0, 1])
        df.columns = ["label", "text"]
    else:
        kwargs = dict(sep="\t", header=None, names=["label", "text"], quoting=csv.QUOTE_NONE)
        try:
            df = pd.read_csv(path, encoding="utf-8", **kwargs)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1", **kwargs)

    df = df.dropna(subset=["label", "text"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)
    df["text"] = df["text"].astype(str)
    df["label_num"] = (df["label"] == "spam").astype(int)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=None, help="load this file instead of downloading")
    args = ap.parse_args()
    data = load_raw(args.path)
    print(data.head())
    print(f"\n{len(data)} messages\n{data['label'].value_counts().to_string()}")
