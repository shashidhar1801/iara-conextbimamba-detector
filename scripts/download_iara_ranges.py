#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import requests
from tqdm import tqdm


ZENODO_RECORD_API = "https://zenodo.org/api/records/15777429"


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def expected_md5(meta: dict) -> str | None:
    checksum = meta.get("checksum", "")
    if checksum.startswith("md5:"):
        return checksum.split(":", 1)[1].lower()
    return None


def stream_file(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    mode = "ab" if existing else "wb"
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + existing
        with tqdm(total=total, initial=existing, unit="B", unit_scale=True, desc=out_path.name) as bar:
            with tmp.open(mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
    tmp.replace(out_path)


def download_range(
    url: str,
    part_path: Path,
    range_start: int,
    range_end: int,
    bar: tqdm,
    lock: Lock,
) -> None:
    expected_size = range_end - range_start + 1
    reported = 0
    last_error: Exception | None = None
    for attempt in range(8):
        existing = part_path.stat().st_size if part_path.exists() else 0
        if existing == expected_size:
            with lock:
                bar.update(existing - reported)
            return
        if existing > expected_size:
            part_path.unlink()
            existing = 0
            reported = 0

        with lock:
            bar.update(existing - reported)
        reported = existing

        current_start = range_start + existing
        headers = {"Range": f"bytes={current_start}-{range_end}"}
        mode = "ab" if existing else "wb"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=180) as r:
                r.raise_for_status()
                if r.status_code != 206:
                    raise RuntimeError(f"Expected HTTP 206 for ranged request, got {r.status_code}")
                with part_path.open(mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            reported += len(chunk)
                            with lock:
                                bar.update(len(chunk))
        except Exception as exc:
            last_error = exc
            time.sleep(min(30, 2 * (attempt + 1)))
            continue

    final_size = part_path.stat().st_size if part_path.exists() else 0
    if final_size != expected_size:
        raise RuntimeError(f"Part {part_path} has {final_size} bytes, expected {expected_size}") from last_error


def download_ranged(url: str, out_path: Path, size: int, workers: int, chunk_size: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size == size:
        print(f"Already downloaded: {out_path}")
        return
    if out_path.exists() and out_path.stat().st_size != size:
        out_path.unlink()

    parts_dir = out_path.parent / f".{out_path.name}.parts"
    parts_dir.mkdir(exist_ok=True)
    ranges = []
    num_parts = int(math.ceil(size / chunk_size))
    for idx in range(num_parts):
        start = idx * chunk_size
        end = min(size - 1, start + chunk_size - 1)
        ranges.append((idx, start, end, parts_dir / f"part-{idx:05d}"))

    lock = Lock()
    with tqdm(total=size, unit="B", unit_scale=True, desc=out_path.name) as bar:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(download_range, url, part_path, start, end, bar, lock)
                for idx, start, end, part_path in ranges
            ]
            for future in as_completed(futures):
                future.result()

    tmp = out_path.with_suffix(out_path.suffix + ".joining")
    with tmp.open("wb") as out:
        for _, _, _, part_path in ranges:
            with part_path.open("rb") as part:
                for block in iter(lambda: part.read(16 * 1024 * 1024), b""):
                    out.write(block)
    if tmp.stat().st_size != size:
        raise RuntimeError(f"Combined file size mismatch for {tmp}")
    tmp.replace(out_path)

    for _, _, _, part_path in ranges:
        part_path.unlink()
    parts_dir.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel ranged downloader for IARA v2 Zenodo files.")
    parser.add_argument("--out-dir", default="data/iara_raw")
    parser.add_argument("--files", nargs="*", default=["iara.xlsx"], help="File names, or 'all'.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mb", type=int, default=256)
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
        size = int(meta["size"])
        print(f"Downloading {name} ({size / 1e9:.2f} GB) -> {out_path}")
        if size < 100 * 1024 * 1024:
            if not (out_path.exists() and out_path.stat().st_size == size):
                stream_file(url, out_path)
            else:
                print(f"Already downloaded: {out_path}")
        else:
            download_ranged(
                url=url,
                out_path=out_path,
                size=size,
                workers=max(1, args.workers),
                chunk_size=max(16, args.chunk_mb) * 1024 * 1024,
            )

        want_md5 = expected_md5(meta)
        if want_md5 and not args.skip_md5:
            got_md5 = md5sum(out_path)
            if got_md5.lower() != want_md5:
                raise RuntimeError(f"MD5 mismatch for {out_path}: expected {want_md5}, got {got_md5}")
            print(f"MD5 OK: {out_path.name}")


if __name__ == "__main__":
    main()
