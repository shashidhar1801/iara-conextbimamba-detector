#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from iara_conextbimamba.data import IARASegmentDataset
from iara_conextbimamba.metrics import binary_metrics
from iara_conextbimamba.model import build_model
from iara_conextbimamba.utils import ensure_dir, move_to_device, save_json


def build_dataset(cfg, split: str) -> IARASegmentDataset:
    data = cfg["data"]
    return IARASegmentDataset(
        manifest_csv=data["manifest_csv"],
        split=split,
        split_column=data.get("split_column", "split"),
        path_column=data.get("path_column", "audio_path"),
        target_column=data.get("target_column", "target"),
        recording_id_column=data.get("recording_id_column", "recording_id"),
        sample_rate=int(data.get("sample_rate", 16000)),
        segment_seconds=float(data.get("segment_seconds", 10.0)),
        hop_seconds=float(data.get("hop_seconds", 5.0)),
        min_segment_seconds=float(data.get("min_segment_seconds", 2.0)),
        n_fft=int(data.get("n_fft", 1024)),
        hop_length=int(data.get("hop_length", 512)),
        win_length=int(data.get("win_length", data.get("n_fft", 1024))),
        n_mels=int(data.get("n_mels", 128)),
        f_min=float(data.get("f_min", 10.0)),
        f_max=float(data.get("f_max", data.get("sample_rate", 16000) / 2)),
        feature_mode=data.get("feature_mode", "logmel"),
        frontend_config=data.get("multi_resolution", {}),
        augment=False,
        augmentation_config=cfg.get("augmentation", {}),
    )


@torch.no_grad()
def run_eval(model, loader, device, threshold: float):
    model.eval()
    rows = []
    y_true, y_prob = [], []
    losses = []
    for batch in tqdm(loader, desc="eval", leave=False):
        features = batch["features"].to(device, non_blocking=True)
        out = model(features)
        logits = out["logits"]
        target_dev = batch["target"].to(device, non_blocking=True)
        probs = torch.sigmoid(logits).cpu().numpy()
        targets = batch["target"].numpy()
        losses.append(float(F.binary_cross_entropy_with_logits(logits, target_dev).detach().cpu()))
        y_true.extend(targets.tolist())
        y_prob.extend(probs.tolist())

        for i in range(len(probs)):
            rows.append(
                {
                    "recording_id": batch["recording_id"][i],
                    "audio_path": batch["audio_path"][i],
                    "start_sec": float(batch["start_sec"][i]),
                    "duration_sec": float(batch["duration_sec"][i]),
                    "target": int(targets[i]),
                    "ship_probability": float(probs[i]),
                    "prediction": int(probs[i] >= threshold),
                }
            )

    segment_metrics = binary_metrics(y_true, y_prob, threshold=threshold, prefix="segment_")
    segment_metrics["segment_loss"] = float(sum(losses) / max(len(losses), 1))

    seg_df = pd.DataFrame(rows)
    rec_df = (
        seg_df.groupby("recording_id")
        .agg(
            audio_path=("audio_path", "first"),
            target=("target", "first"),
            ship_probability=("ship_probability", "mean"),
            n_segments=("ship_probability", "size"),
        )
        .reset_index()
    )
    rec_df["prediction"] = (rec_df["ship_probability"] >= threshold).astype(int)
    recording_metrics = binary_metrics(rec_df["target"], rec_df["ship_probability"], threshold=threshold, prefix="recording_")
    return seg_df, rec_df, {**segment_metrics, **recording_metrics}


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained Underwater ConExtBiMamba checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    threshold = args.threshold if args.threshold is not None else float(cfg["training"].get("threshold", 0.5))

    model = build_model(cfg["model"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device)

    ds = build_dataset(cfg, args.split)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["data"].get("batch_size", 16)),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=bool(cfg["data"].get("pin_memory", True)),
    )

    out_dir = ensure_dir(args.out_dir or str(Path(args.checkpoint).parent / f"eval_{args.split}"))
    seg_df, rec_df, metrics = run_eval(model, loader, device, threshold)
    seg_df.to_csv(out_dir / "segment_predictions.csv", index=False)
    rec_df.to_csv(out_dir / "recording_predictions.csv", index=False)
    save_json(metrics, out_dir / "metrics.json")

    print(f"Wrote predictions and metrics to {out_dir}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
