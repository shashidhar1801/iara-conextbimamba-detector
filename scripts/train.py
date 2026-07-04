#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Allow running from the project root without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from iara_conextbimamba.data import IARASegmentDataset
from iara_conextbimamba.metrics import binary_metrics
from iara_conextbimamba.model import build_model
from iara_conextbimamba.utils import count_parameters, ensure_dir, load_yaml, move_to_device, save_json, set_seed


def build_dataset(cfg, split: str, augment: bool) -> IARASegmentDataset:
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
        augment=augment,
        augmentation_config=cfg.get("augmentation", {}),
    )


def build_loader(cfg, split: str, augment: bool, shuffle: bool) -> DataLoader:
    ds = build_dataset(cfg, split, augment=augment)
    data = cfg["data"]
    return DataLoader(
        ds,
        batch_size=int(data.get("batch_size", 16)),
        shuffle=shuffle,
        num_workers=int(data.get("num_workers", 4)),
        pin_memory=bool(data.get("pin_memory", True)),
        drop_last=False,
    )


def infer_pos_weight(manifest_csv: str, split_column: str = "split", split: str = "train", target_column: str = "target") -> float:
    df = pd.read_csv(manifest_csv)
    if split_column in df.columns:
        df = df[df[split_column].astype(str) == split]
    pos = float((df[target_column] == 1).sum())
    neg = float((df[target_column] == 0).sum())
    return neg / max(pos, 1.0)


def train_epoch(model, loader, optimizer, scaler, device, cfg, pos_weight_tensor):
    model.train()
    size_loss_weight = float(cfg["training"].get("size_loss_weight", 0.25))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))
    amp_enabled = bool(cfg["training"].get("amp", True)) and device.type == "cuda"

    losses = []
    y_true, y_prob = [], []
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        optimizer.zero_grad(set_to_none=True)
        features = batch["features"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            out = model(features)
            logits = out["logits"]
            target = batch["target"].to(device, non_blocking=True)
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight_tensor)
            if "size_logits" in out and "size_label" in batch:
                size_label = batch["size_label"].to(device, non_blocking=True)
                valid = size_label >= 0
                if valid.any():
                    loss = loss + size_loss_weight * F.cross_entropy(out["size_logits"][valid], size_label[valid])

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.detach().cpu()))
        probs = torch.sigmoid(logits.detach()).cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(target.detach().cpu().numpy().tolist())
        pbar.set_postfix(loss=np.mean(losses))

    metrics = binary_metrics(y_true, y_prob, threshold=float(cfg["training"].get("threshold", 0.5)), prefix="train_")
    metrics["train_loss"] = float(np.mean(losses))
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, cfg, prefix="val_"):
    model.eval()
    losses = []
    y_true, y_prob = [], []
    pbar = tqdm(loader, desc=prefix.rstrip("_"), leave=False)
    for batch in pbar:
        features = batch["features"].to(device, non_blocking=True)
        out = model(features)
        logits = out["logits"]
        target = batch["target"].to(device, non_blocking=True)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        losses.append(float(loss.detach().cpu()))
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(target.detach().cpu().numpy().tolist())
    metrics = binary_metrics(y_true, y_prob, threshold=float(cfg["training"].get("threshold", 0.5)), prefix=prefix)
    metrics[f"{prefix}loss"] = float(np.mean(losses))
    return metrics


def save_checkpoint(path: Path, model, optimizer, epoch: int, cfg, metrics, best_score):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": cfg,
            "metrics": metrics,
            "best_score": best_score,
            "model_backend": model.backend_summary(),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Underwater ConExtBiMamba on IARA.")
    parser.add_argument("--config", default="configs/iara_binary.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["project"].get("seed", 42)))
    out_dir = ensure_dir(cfg["project"]["output_dir"])
    save_json(cfg, out_dir / "config_resolved.json")

    train_loader = build_loader(cfg, "train", augment=True, shuffle=True)
    val_loader = build_loader(cfg, "val", augment=False, shuffle=False)

    model = build_model(cfg["model"])
    device = torch.device(args.device)
    model.to(device)

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Mamba backend: {model.backend_summary()}")
    print(f"Train segments: {len(train_loader.dataset):,}; Val segments: {len(val_loader.dataset):,}")

    pos_weight_cfg = cfg["training"].get("pos_weight", "auto")
    if str(pos_weight_cfg).lower() == "auto":
        pos_weight = infer_pos_weight(
            cfg["data"]["manifest_csv"],
            split_column=cfg["data"].get("split_column", "split"),
            split="train",
            target_column=cfg["data"].get("target_column", "target"),
        )
    else:
        pos_weight = float(pos_weight_cfg)
    pos_weight_tensor = torch.tensor(pos_weight, dtype=torch.float32, device=device)
    print(f"BCE positive class weight: {pos_weight:.3f}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"].get("learning_rate", 3e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 1e-6)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and bool(cfg["training"].get("amp", True))))

    monitor = cfg["training"].get("monitor", "val_balanced_accuracy")
    mode = cfg["training"].get("monitor_mode", "max")
    best_score = -float("inf") if mode == "max" else float("inf")
    best_epoch = -1
    patience = int(cfg["training"].get("early_stop_patience", 12))
    history = []

    for epoch in range(1, int(cfg["training"].get("epochs", 60)) + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, scaler, device, cfg, pos_weight_tensor)
        val_metrics = evaluate(model, val_loader, device, cfg, prefix="val_")
        metrics = {"epoch": epoch, **train_metrics, **val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        history.append(metrics)

        score = metrics.get(monitor, float("nan"))
        scheduler.step(score if np.isfinite(score) else 0.0)
        improved = (score > best_score) if mode == "max" else (score < best_score)
        if improved:
            best_score = score
            best_epoch = epoch
            save_checkpoint(out_dir / "best.pt", model, optimizer, epoch, cfg, metrics, best_score)
        save_checkpoint(out_dir / "last.pt", model, optimizer, epoch, cfg, metrics, best_score)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

        print(
            f"Epoch {epoch:03d} | train_loss={metrics['train_loss']:.4f} "
            f"val_loss={metrics['val_loss']:.4f} val_bal_acc={metrics['val_balanced_accuracy']:.4f} "
            f"val_f1={metrics['val_f1']:.4f} val_auroc={metrics['val_auroc']:.4f}"
        )

        if epoch - best_epoch >= patience:
            print(f"Early stopping: no improvement in {patience} epochs. Best epoch={best_epoch} score={best_score:.4f}")
            break

    print(f"Training complete. Best checkpoint: {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
