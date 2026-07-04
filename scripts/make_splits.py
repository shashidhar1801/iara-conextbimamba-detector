#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def choose_stratify(df: pd.DataFrame, preferred: str) -> pd.Series:
    if preferred in df.columns:
        s = df[preferred].astype(str)
        if s.value_counts().min() >= 2:
            return s
    for col in ["target", "label", "dc", "platform"]:
        if col in df.columns:
            s = df[col].astype(str)
            if s.value_counts().min() >= 2:
                return s
    return pd.Series(["all"] * len(df), index=df.index)


def safe_train_test_split(indices, test_size, stratify, seed):
    try:
        return train_test_split(indices, test_size=test_size, random_state=seed, stratify=stratify)
    except ValueError:
        return train_test_split(indices, test_size=test_size, random_state=seed, stratify=None)


def main():
    parser = argparse.ArgumentParser(description="Create recording-level train/val/test splits for IARA.")
    parser.add_argument("--manifest", default="data/iara_manifest.csv")
    parser.add_argument("--out", default="data/iara_manifest_splits.csv")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify", default="dc", help="Preferred stratification column, e.g. dc, platform, label, size_name.")
    parser.add_argument("--split-mode", choices=["random", "uo_to_glider"], default="random",
                        help="random = recording-level stratified split; uo_to_glider = train/val on UO A-E and test on glider F-H.")
    args = parser.parse_args()

    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError("--train + --val + --test must equal 1.0")

    df = pd.read_csv(args.manifest)
    if "recording_id" not in df.columns:
        df["recording_id"] = df["audio_path"].apply(lambda p: Path(p).stem)

    # One row per recording, so splitting rows is a recording-level split.
    if args.split_mode == "uo_to_glider":
        if "platform" in df.columns:
            is_glider = df["platform"].astype(str).str.lower().eq("glider")
        elif "dc" in df.columns:
            is_glider = df["dc"].astype(str).str.upper().isin(["F", "G", "H"])
        else:
            raise ValueError("uo_to_glider split requires either platform or dc column.")
        test_idx = df.index[is_glider].to_numpy()
        trainval_idx = df.index[~is_glider].to_numpy()
        if len(test_idx) == 0 or len(trainval_idx) == 0:
            raise ValueError("uo_to_glider split produced an empty train/val or test set.")
        trainval_df = df.loc[trainval_idx]
        stratify_tv = choose_stratify(trainval_df, args.stratify)
        train_idx, val_idx = safe_train_test_split(trainval_idx, test_size=args.val / max(args.train + args.val, 1e-12), stratify=stratify_tv, seed=args.seed)
    else:
        idx = df.index.to_numpy()
        stratify = choose_stratify(df, args.stratify)
        train_idx, temp_idx = safe_train_test_split(idx, test_size=args.val + args.test, stratify=stratify, seed=args.seed)

        temp_df = df.loc[temp_idx]
        temp_stratify = choose_stratify(temp_df, args.stratify)
        relative_test = args.test / (args.val + args.test)
        val_idx, test_idx = safe_train_test_split(temp_idx, test_size=relative_test, stratify=temp_stratify, seed=args.seed)

    df["split"] = ""
    df.loc[train_idx, "split"] = "train"
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"Wrote {len(df):,} recordings to {out}")
    print("\nSplit counts:")
    print(df["split"].value_counts().to_string())
    if "label" in df.columns:
        print("\nSplit x label:")
        print(pd.crosstab(df["split"], df["label"]).to_string())
    if "dc" in df.columns:
        print("\nSplit x DC:")
        print(pd.crosstab(df["split"], df["dc"]).to_string())


if __name__ == "__main__":
    main()
