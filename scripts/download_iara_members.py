#!/usr/bin/env python
from __future__ import annotations

import argparse
import http.client
import json
import struct
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import pandas as pd
from tqdm import tqdm


ZENODO_RECORD_API = "https://zenodo.org/api/records/15777429"


def http_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def http_range(url: str, start: int, end: int) -> bytes:
    last_error: Exception | None = None
    expected = end - start + 1
    for attempt in range(5):
        try:
            request = Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urlopen(request, timeout=120) as response:
                data = response.read()
            if len(data) != expected:
                raise http.client.IncompleteRead(data, expected - len(data))
            return data
        except Exception as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    assert last_error is not None
    raise last_error


def zip64_values(extra: bytes) -> list[int]:
    values: list[int] = []
    i = 0
    while i + 4 <= len(extra):
        header_id, size = struct.unpack("<HH", extra[i : i + 4])
        data = extra[i + 4 : i + 4 + size]
        if header_id == 0x0001:
            for j in range(0, len(data), 8):
                if j + 8 <= len(data):
                    values.append(struct.unpack("<Q", data[j : j + 8])[0])
        i += 4 + size
    return values


def central_directory(url: str, size: int) -> bytes:
    tail = http_range(url, max(0, size - 65536), size - 1)
    eocd = tail.rfind(b"PK\x05\x06")
    if eocd == -1:
        raise RuntimeError("Could not find ZIP EOCD")
    fields = struct.unpack("<4s4H2LH", tail[eocd : eocd + 22])
    entries, cd_size, cd_offset = fields[4], fields[5], fields[6]
    if entries == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF:
        locator = tail.rfind(b"PK\x06\x07")
        if locator == -1:
            raise RuntimeError("Could not find ZIP64 locator")
        _, _, eocd64_offset, _ = struct.unpack("<4sLQL", tail[locator : locator + 20])
        end = http_range(url, eocd64_offset, size - 1)
        fields64 = struct.unpack("<4sQ2H2L4Q", end[:56])
        cd_size, cd_offset = fields64[8], fields64[9]
    return http_range(url, cd_offset, cd_offset + cd_size - 1)


def parse_central_directory(cd: bytes) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    i = 0
    while i < len(cd) - 46:
        if cd[i : i + 4] != b"PK\x01\x02":
            i += 1
            continue
        fields = struct.unpack("<4s6H3L5H2L", cd[i : i + 46])
        method = fields[4]
        crc = fields[7]
        comp_size = fields[8]
        uncomp_size = fields[9]
        fn_len = fields[10]
        extra_len = fields[11]
        comment_len = fields[12]
        local_offset = fields[16]
        name = cd[i + 46 : i + 46 + fn_len].decode("utf-8")
        extra = cd[i + 46 + fn_len : i + 46 + fn_len + extra_len]
        z64 = zip64_values(extra)
        z64_i = 0
        if uncomp_size == 0xFFFFFFFF:
            uncomp_size = z64[z64_i]
            z64_i += 1
        if comp_size == 0xFFFFFFFF:
            comp_size = z64[z64_i]
            z64_i += 1
        if local_offset == 0xFFFFFFFF:
            local_offset = z64[z64_i]
        entries[name] = {
            "name": name,
            "method": method,
            "crc": crc,
            "comp_size": comp_size,
            "uncomp_size": uncomp_size,
            "local_offset": local_offset,
        }
        i += 46 + fn_len + extra_len + comment_len
    return entries


def fetch_member(url: str, member: dict[str, Any], out_path: Path) -> None:
    if out_path.exists() and out_path.stat().st_size == int(member["uncomp_size"]):
        return
    local_offset = int(member["local_offset"])
    local_header = http_range(url, local_offset, local_offset + 30 - 1)
    if local_header[:4] != b"PK\x03\x04":
        raise RuntimeError(f"Bad local header for {member['name']}")
    fields = struct.unpack("<4s5H3L2H", local_header)
    fn_len = fields[9]
    extra_len = fields[10]
    data_start = local_offset + 30 + fn_len + extra_len
    compressed = http_range(url, data_start, data_start + int(member["comp_size"]) - 1)
    method = int(member["method"])
    if method == 0:
        payload = compressed
    elif method == 8:
        payload = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"Unsupported ZIP compression method {method} for {member['name']}")
    if len(payload) != int(member["uncomp_size"]):
        raise RuntimeError(f"Size mismatch for {member['name']}: got {len(payload)}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download selected IARA WAV members from Zenodo ZIPs by byte range.")
    parser.add_argument("--manifest", required=True, help="CSV with archive_member column.")
    parser.add_argument("--out-dir", default="data/iara_subset")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    members = sorted(df["archive_member"].dropna().astype(str).unique())
    if not members:
        raise ValueError("No archive_member values found.")

    record = http_json(ZENODO_RECORD_API)
    files = {item["key"]: item for item in record["files"]}
    out_dir = Path(args.out_dir)

    by_archive: dict[str, list[str]] = {}
    for member in members:
        archive = member.split("/", 1)[0] + ".zip"
        by_archive.setdefault(archive, []).append(member)

    for archive, archive_members in by_archive.items():
        meta = files[archive]
        url = meta["links"]["self"]
        size = int(meta["size"])
        print(f"Indexing {archive} for {len(archive_members)} member(s)")
        entries = parse_central_directory(central_directory(url, size))
        missing = [name for name in archive_members if name not in entries]
        if missing:
            raise ValueError(f"{archive} missing requested members: {missing[:5]}")

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [
                pool.submit(fetch_member, url, entries[name], out_dir / name)
                for name in archive_members
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc=archive):
                future.result()

    print(f"Downloaded {len(members)} WAV files to {out_dir}")


if __name__ == "__main__":
    main()
