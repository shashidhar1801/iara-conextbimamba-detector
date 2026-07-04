#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm

ZENODO_RECORD_API = "https://zenodo.org/api/records/15777429"  # IARA v2


def md5sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {}
    mode = "wb"
    existing = out_path.stat().st_size if out_path.exists() else 0
    if existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"

    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + (existing if mode == "ab" else 0)
        with tqdm(total=total, initial=(existing if mode == "ab" else 0), unit="B", unit_scale=True, desc=out_path.name) as pbar:
            with out_path.open(mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


def main():
    parser = argparse.ArgumentParser(description="Download IARA v2 from Zenodo.")
    parser.add_argument("--out-dir", default="data/iara_raw")
    parser.add_argument("--files", nargs="*", default=["iara.xlsx"], help="Files to download. Use A.zip ... H.zip iara.xlsx, or 'all'.")
    parser.add_argument("--extract", action="store_true", help="Extract zip files after download.")
    parser.add_argument("--skip-md5", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    record = requests.get(ZENODO_RECORD_API, timeout=60).json()
    available = {f["key"]: f for f in record["files"]}
    requested = list(available) if args.files == ["all"] or "all" in args.files else args.files

    for name in requested:
        if name not in available:
            print(f"Unknown file '{name}'. Available: {sorted(available)}", file=sys.stderr)
            sys.exit(2)
        meta = available[name]
        url = meta["links"]["self"]
        out_path = out_dir / name
        print(f"Downloading {name} -> {out_path}")
        download_file(url, out_path)

        expected_md5 = None
        checksum = meta.get("checksum", "")
        if checksum.startswith("md5:"):
            expected_md5 = checksum.split(":", 1)[1]
        if expected_md5 and not args.skip_md5:
            got = md5sum(out_path)
            if got.lower() != expected_md5.lower():
                raise RuntimeError(f"MD5 mismatch for {out_path}: expected {expected_md5}, got {got}")
            print(f"MD5 OK: {out_path.name}")

        if args.extract and out_path.suffix.lower() == ".zip":
            extract_dir = out_dir / out_path.stem
            extract_dir.mkdir(exist_ok=True)
            print(f"Extracting {out_path.name} -> {extract_dir}")
            with ZipFile(out_path) as zf:
                zf.extractall(extract_dir)


if __name__ == "__main__":
    main()
