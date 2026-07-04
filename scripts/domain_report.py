#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from iara_conextbimamba.metrics import binary_metrics
from iara_conextbimamba.utils import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Report IARA ship-detection metrics by domain/metadata group.")
    parser.add_argument("--predictions", required=True, help="recording_predictions.csv from scripts/evaluate.py")
    parser.add_argument("--manifest", required=True, help="Prepared manifest CSV with metadata columns.")
    parser.add_argument("--out", default=None, help="Output CSV for grouped metrics.")
    parser.add_argument(
        "--groups",
        nargs="*",
        default=["dc", "platform", "cpa", "range_bin", "size_name", "vessel_type"],
        help="Metadata columns to group by.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    pred = pd.read_csv(args.predictions)
    manifest = pd.read_csv(args.manifest)
    if "recording_id" not in pred.columns or "recording_id" not in manifest.columns:
        raise ValueError("Both predictions and manifest must contain recording_id.")

    merged = pred.merge(
        manifest.drop_duplicates("recording_id"),
        on="recording_id",
        how="left",
        suffixes=("", "_manifest"),
    )
    target_col = "target" if "target" in merged.columns else "target_manifest"
    score_col = "ship_probability"
    if target_col not in merged.columns or score_col not in merged.columns:
        raise ValueError("Predictions must contain ship_probability and target columns, or manifest must contain target.")

    rows = []
    for group_col in args.groups:
        if group_col not in merged.columns:
            continue
        for value, sub in merged.groupby(group_col, dropna=False):
            if len(sub) == 0:
                continue
            m = binary_metrics(sub[target_col], sub[score_col], threshold=args.threshold, prefix="")
            rows.append({"group_column": group_col, "group_value": value, "n": len(sub), **m})

    out_df = pd.DataFrame(rows)
    print(out_df.to_string(index=False) if not out_df.empty else "No requested group columns found.")
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out, index=False)
        save_json({"groups": args.groups, "threshold": args.threshold}, out.with_suffix(".json"))
        print(f"Wrote domain report to {out}")


if __name__ == "__main__":
    main()
