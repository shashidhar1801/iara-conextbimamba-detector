#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from iara_conextbimamba.features import LogMelExtractor, segment_cmvn
from iara_conextbimamba.model import UnderwaterConExtBiMambaDetector


def load_audio(path: str, target_sr: int) -> torch.Tensor:
    try:
        wav, sr = torchaudio.load(path)
    except Exception:
        data, sr = sf.read(path, always_2d=True)
        wav = torch.from_numpy(data.T.astype("float32"))
    wav = wav.mean(dim=0).float()
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=int(sr), new_freq=target_sr)
    wav = wav - wav.mean()
    wav = wav / wav.abs().max().clamp_min(1e-6)
    return wav


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Run ship/background detection on one underwater audio file.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    threshold = args.threshold if args.threshold is not None else float(cfg["training"].get("threshold", 0.5))
    data_cfg = cfg["data"]
    sample_rate = int(data_cfg.get("sample_rate", 16000))
    segment_seconds = float(data_cfg.get("segment_seconds", 10.0))
    hop_seconds = float(data_cfg.get("hop_seconds", 5.0))

    device = torch.device(args.device)
    model = UnderwaterConExtBiMambaDetector(**cfg["model"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    frontend = LogMelExtractor(
        sample_rate=sample_rate,
        n_fft=int(data_cfg.get("n_fft", 1024)),
        hop_length=int(data_cfg.get("hop_length", 512)),
        win_length=int(data_cfg.get("win_length", data_cfg.get("n_fft", 1024))),
        n_mels=int(data_cfg.get("n_mels", 128)),
        f_min=float(data_cfg.get("f_min", 10.0)),
        f_max=float(data_cfg.get("f_max", sample_rate / 2)),
    ).to(device)

    wav = load_audio(args.audio, sample_rate).to(device)
    seg_len = int(round(segment_seconds * sample_rate))
    hop_len = int(round(hop_seconds * sample_rate))
    starts = list(range(0, max(1, wav.numel() - seg_len + 1), hop_len))
    if not starts or starts[-1] + seg_len < wav.numel():
        starts.append(max(0, wav.numel() - seg_len))

    rows = []
    for start in tqdm(starts, desc="segments"):
        seg = wav[start : start + seg_len]
        if seg.numel() < seg_len:
            seg = torch.nn.functional.pad(seg, (0, seg_len - seg.numel()))
        feats = segment_cmvn(frontend(seg)).unsqueeze(0)
        prob = torch.sigmoid(model(feats)["logits"])[0].item()
        rows.append({"start_sec": start / sample_rate, "ship_probability": prob, "prediction": int(prob >= threshold)})

    df = pd.DataFrame(rows).drop_duplicates(subset=["start_sec"])
    recording_prob = float(df["ship_probability"].mean())
    recording_pred = int(recording_prob >= threshold)
    print(f"Recording ship probability: {recording_prob:.4f}; prediction: {'ship' if recording_pred else 'background'}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Wrote segment predictions to {out}")


if __name__ == "__main__":
    main()
