#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from iara_conextbimamba.data import build_manifest


def main():
    parser = argparse.ArgumentParser(description="Create a recording-level IARA manifest from extracted audio and iara.xlsx.")
    parser.add_argument("--audio-root", required=True, help="Directory containing extracted IARA folders A-H.")
    parser.add_argument("--excel", default=None, help="Path to iara.xlsx. Optional, but recommended.")
    parser.add_argument("--out", default="data/iara_manifest.csv")
    parser.add_argument("--filename-col", default=None, help="Override metadata filename column if auto-detection fails.")
    parser.add_argument("--dc-col", default=None, help="Override metadata data-collection/DC column if auto-detection fails.")
    parser.add_argument("--label-col", default=None, help="Override metadata label column if auto-detection fails.")
    parser.add_argument("--size-col", default=None, help="Override metadata vessel-size column if auto-detection fails.")
    parser.add_argument("--vessel-type-col", default=None, help="Override metadata vessel-type column if auto-detection fails.")
    args = parser.parse_args()

    manifest = build_manifest(
        audio_root=args.audio_root,
        excel_path=args.excel,
        filename_col=args.filename_col,
        dc_col=args.dc_col,
        label_col=args.label_col,
        size_col=args.size_col,
        vessel_type_col=args.vessel_type_col,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)

    print(f"Wrote {len(manifest):,} recordings to {out}")
    print("\nLabel distribution:")
    print(manifest["label"].value_counts(dropna=False).to_string())
    print("\nDC distribution:")
    print(manifest["dc"].value_counts(dropna=False).sort_index().to_string())
    print("\nPlatform x label:")
    print(pd.crosstab(manifest["platform"], manifest["label"]).to_string())


if __name__ == "__main__":
    main()
