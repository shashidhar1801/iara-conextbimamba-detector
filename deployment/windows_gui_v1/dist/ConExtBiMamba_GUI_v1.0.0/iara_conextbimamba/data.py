from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from .features import (
    LogMelExtractor,
    MultiResolutionAcousticExtractor,
    add_gaussian_noise,
    freq_mask,
    random_gain,
    segment_cmvn,
    time_mask,
)

AUDIO_EXTENSIONS = {".wav", ".flac", ".aiff", ".aif", ".ogg", ".mp3"}
SHIP_DCS = {"A", "B", "C", "D", "F", "G"}
BACKGROUND_DCS = {"E", "H"}
ALL_DCS = SHIP_DCS | BACKGROUND_DCS

DC_INFO = {
    "A": {"platform": "uo", "target": 1, "cpa": 1, "range_bin": "lt250m"},
    "B": {"platform": "uo", "target": 1, "cpa": 0, "range_bin": "lt250m"},
    "C": {"platform": "uo", "target": 1, "cpa": 1, "range_bin": "250_500m"},
    "D": {"platform": "uo", "target": 1, "cpa": 0, "range_bin": "250_500m"},
    "E": {"platform": "uo", "target": 0, "cpa": -1, "range_bin": "background"},
    "F": {"platform": "glider", "target": 1, "cpa": -1, "range_bin": "glider"},
    "G": {"platform": "glider", "target": 1, "cpa": -1, "range_bin": "glider"},
    "H": {"platform": "glider", "target": 0, "cpa": -1, "range_bin": "background"},
}


def clean_column_name(name: object) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_") or "unnamed"


def discover_audio_files(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    records = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
            dc = infer_dc_from_path(path)
            records.append(
                {
                    "audio_path": str(path.resolve()),
                    "filename": path.name,
                    "stem": path.stem.lower(),
                    "dc_from_path": dc,
                }
            )
    if not records:
        raise FileNotFoundError(f"No audio files found under {root}")
    return pd.DataFrame(records)


def infer_dc_from_path(path: str | Path) -> str:
    p = Path(path)
    parts = [part.upper() for part in p.parts]
    for part in reversed(parts):
        if part in ALL_DCS:
            return part
    stem = p.stem.upper()
    m = re.match(r"^([A-H])([_\-]|\d|$)", stem)
    if m:
        return m.group(1)
    return ""


def read_iara_excel(excel_path: str | Path) -> pd.DataFrame:
    excel_path = Path(excel_path)
    sheets = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")
    frames = []
    for sheet_name, df in sheets.items():
        if df.empty:
            continue
        df = df.copy()
        df.columns = [clean_column_name(c) for c in df.columns]
        df["metadata_sheet"] = sheet_name
        frames.append(df)
    if not frames:
        raise ValueError(f"No non-empty sheets found in {excel_path}")
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(axis=1, how="all")
    return out


def _find_first_col(columns: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    cols = list(columns)
    for cand in candidates:
        if cand in cols:
            return cand
    for cand in candidates:
        for col in cols:
            if cand in col:
                return col
    return None


def infer_metadata_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    cols = list(df.columns)
    return {
        "filename": _find_first_col(cols, ["filename", "file_name", "file", "audio", "recording", "wav", "name"]),
        "dc": _find_first_col(cols, ["dc", "data_collection", "collection", "dataset", "subset", "group"]),
        "label": _find_first_col(cols, ["label", "target", "class", "category", "type", "noise_type"]),
        "size": _find_first_col(cols, ["size_class", "ship_size", "vessel_size", "size", "length_class"]),
        "vessel_type": _find_first_col(cols, ["vessel_type", "ship_type", "type_vessel", "ais_type"]),
    }


def _normalize_key(value: object) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    p = Path(s)
    return p.stem.lower()


def _infer_target_from_value(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s in {"1", "true", "ship", "vessel", "target", "ship_noise", "ship noise", "vessel_noise", "vessel noise"}:
        return 1
    if s in {"0", "false", "background", "bg", "ambient", "noise", "background_noise", "background noise"}:
        return 0
    if "background" in s or s in {"ambient noise", "non_ship", "no_ship"}:
        return 0
    if "ship" in s or "vessel" in s or "target" in s:
        return 1
    return None


def _infer_size_label(value: object) -> int:
    if pd.isna(value):
        return -1
    s = str(value).strip().lower()
    if "background" in s or s in {"bg", "noise", "ambient"}:
        return 0
    if "small" in s or s in {"s", "1"}:
        return 1
    if "medium" in s or s in {"m", "2"}:
        return 2
    if "large" in s or s in {"l", "3"}:
        return 3
    return -1


def build_manifest(
    audio_root: str | Path,
    excel_path: Optional[str | Path] = None,
    filename_col: Optional[str] = None,
    dc_col: Optional[str] = None,
    label_col: Optional[str] = None,
    size_col: Optional[str] = None,
    vessel_type_col: Optional[str] = None,
) -> pd.DataFrame:
    """Build a recording-level manifest for IARA.

    The function is intentionally permissive because metadata column names can change between
    dataset versions. If labels are unavailable, it infers ship/background from IARA DCs:
    A/B/C/D/F/G = ship and E/H = background.
    """
    audio_df = discover_audio_files(audio_root)
    audio_df["join_key"] = audio_df["stem"].map(_normalize_key)

    if excel_path is not None:
        meta = read_iara_excel(excel_path)
        inferred = infer_metadata_columns(meta)
        filename_col = filename_col or inferred["filename"]
        dc_col = dc_col or inferred["dc"]
        label_col = label_col or inferred["label"]
        size_col = size_col or inferred["size"]
        vessel_type_col = vessel_type_col or inferred["vessel_type"]

        if filename_col and filename_col in meta.columns:
            meta = meta.copy()
            meta["join_key"] = meta[filename_col].map(_normalize_key)
            merged = audio_df.merge(meta, on="join_key", how="left", suffixes=("", "_meta"))
        else:
            merged = audio_df.copy()
            for col in meta.columns:
                merged[col] = np.nan
    else:
        merged = audio_df.copy()
        dc_col = label_col = size_col = vessel_type_col = None

    dc_values = []
    for _, row in merged.iterrows():
        dc = ""
        if dc_col and dc_col in merged.columns and not pd.isna(row.get(dc_col)):
            candidate = str(row.get(dc_col)).strip().upper()
            m = re.search(r"[A-H]", candidate)
            if m:
                dc = m.group(0)
        if not dc:
            dc = str(row.get("dc_from_path", "")).upper()
        dc_values.append(dc)
    merged["dc"] = dc_values

    targets = []
    for _, row in merged.iterrows():
        y = None
        if label_col and label_col in merged.columns:
            y = _infer_target_from_value(row.get(label_col))
        if y is None and row["dc"] in DC_INFO:
            y = int(DC_INFO[row["dc"]]["target"])
        if y is None:
            raise ValueError(
                f"Could not infer target for {row.get('audio_path')}. "
                "Pass --label-col or arrange files under IARA DC folders A-H."
            )
        targets.append(y)
    merged["target"] = targets
    merged["label"] = merged["target"].map({1: "ship", 0: "background"})

    merged["platform"] = [DC_INFO.get(dc, {}).get("platform", "unknown") for dc in merged["dc"]]
    merged["cpa"] = [DC_INFO.get(dc, {}).get("cpa", -1) for dc in merged["dc"]]
    merged["range_bin"] = [DC_INFO.get(dc, {}).get("range_bin", "unknown") for dc in merged["dc"]]

    if size_col and size_col in merged.columns:
        merged["size_label"] = merged[size_col].map(_infer_size_label).astype(int)
    else:
        merged["size_label"] = -1
    merged.loc[merged["target"] == 0, "size_label"] = 0
    merged["size_name"] = merged["size_label"].map({0: "background", 1: "small", 2: "medium", 3: "large"}).fillna("unknown")

    if vessel_type_col and vessel_type_col in merged.columns:
        merged["vessel_type"] = merged[vessel_type_col].fillna("unknown").astype(str)
    elif "vessel_type" not in merged.columns:
        merged["vessel_type"] = "unknown"

    durations = []
    sample_rates = []
    frames = []
    for path in merged["audio_path"]:
        try:
            info = sf.info(path)
            sample_rates.append(int(info.samplerate))
            frames.append(int(info.frames))
            durations.append(float(info.frames) / float(info.samplerate))
        except Exception:
            sample_rates.append(-1)
            frames.append(-1)
            durations.append(np.nan)
    merged["orig_sample_rate"] = sample_rates
    merged["num_frames"] = frames
    merged["duration_sec"] = durations

    merged["recording_id"] = [Path(p).stem for p in merged["audio_path"]]

    cols = [
        "recording_id",
        "audio_path",
        "filename",
        "dc",
        "platform",
        "label",
        "target",
        "size_label",
        "size_name",
        "vessel_type",
        "cpa",
        "range_bin",
        "duration_sec",
        "orig_sample_rate",
        "num_frames",
    ]
    other_cols = [c for c in merged.columns if c not in cols and c not in {"join_key", "stem", "dc_from_path"}]
    out = merged[cols + other_cols].drop_duplicates(subset=["audio_path"]).reset_index(drop=True)
    return out


@dataclass(frozen=True)
class Segment:
    row_index: int
    start_sec: float
    duration_sec: float


class IARASegmentDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        split: str,
        split_column: str = "split",
        path_column: str = "audio_path",
        target_column: str = "target",
        recording_id_column: str = "recording_id",
        sample_rate: int = 16000,
        segment_seconds: float = 10.0,
        hop_seconds: float = 5.0,
        min_segment_seconds: float = 2.0,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: int = 1024,
        n_mels: int = 128,
        f_min: float = 10.0,
        f_max: Optional[float] = None,
        feature_mode: str = "logmel",
        frontend_config: Optional[Dict] = None,
        augment: bool = False,
        augmentation_config: Optional[Dict] = None,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.df = pd.read_csv(self.manifest_csv)
        if split_column not in self.df.columns:
            raise ValueError(f"Manifest {manifest_csv} does not contain split column '{split_column}'.")
        self.df = self.df[self.df[split_column].astype(str) == split].reset_index(drop=True)
        if self.df.empty:
            raise ValueError(f"No rows for split='{split}' in {manifest_csv}")

        self.path_column = path_column
        self.target_column = target_column
        self.recording_id_column = recording_id_column
        self.sample_rate = int(sample_rate)
        self.segment_seconds = float(segment_seconds)
        self.hop_seconds = float(hop_seconds)
        self.min_segment_seconds = float(min_segment_seconds)
        self.augment = augment
        self.aug = augmentation_config or {}

        feature_mode = str(feature_mode).lower()
        if feature_mode in {"multi_resolution_logmel_cqt", "multi_resolution", "logmel_cqt"}:
            self.frontend = MultiResolutionAcousticExtractor(
                sample_rate=self.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
                config=frontend_config,
            )
        elif feature_mode == "logmel":
            self.frontend = LogMelExtractor(
                sample_rate=self.sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=n_mels,
                f_min=f_min,
                f_max=f_max,
            )
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        self.segments = self._make_segments()

    def _make_segments(self) -> List[Segment]:
        segments: List[Segment] = []
        for i, row in self.df.iterrows():
            duration = float(row.get("duration_sec", np.nan))
            if not np.isfinite(duration) or duration <= 0:
                try:
                    info = sf.info(row[self.path_column])
                    duration = float(info.frames) / float(info.samplerate)
                except Exception:
                    duration = self.segment_seconds

            if duration < self.min_segment_seconds:
                continue
            if duration <= self.segment_seconds:
                segments.append(Segment(i, 0.0, min(duration, self.segment_seconds)))
                continue
            start = 0.0
            while start + self.min_segment_seconds <= duration:
                segments.append(Segment(i, start, min(self.segment_seconds, duration - start)))
                start += self.hop_seconds
        if not segments:
            raise ValueError("No usable segments created. Check segment_seconds/min_segment_seconds and audio durations.")
        return segments

    def __len__(self) -> int:
        return len(self.segments)

    def _load_audio(self, path: str, start_sec: float) -> torch.Tensor:
        info = sf.info(path)
        orig_sr = int(info.samplerate)
        frame_offset = int(round(start_sec * orig_sr))
        num_frames = int(round(self.segment_seconds * orig_sr))
        try:
            wav, sr = torchaudio.load(path, frame_offset=frame_offset, num_frames=num_frames)
            if sr != orig_sr:
                orig_sr = sr
        except Exception:
            data, sr = sf.read(path, start=frame_offset, frames=num_frames, always_2d=True)
            wav = torch.from_numpy(data.T.astype(np.float32))
            orig_sr = int(sr)
        wav = wav.mean(dim=0)
        if orig_sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, orig_freq=orig_sr, new_freq=self.sample_rate)
        target_len = int(round(self.segment_seconds * self.sample_rate))
        if wav.numel() < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.numel()))
        elif wav.numel() > target_len:
            wav = wav[:target_len]
        wav = wav.float()
        wav = wav - wav.mean()
        peak = wav.abs().max().clamp_min(1e-6)
        wav = wav / peak
        return wav

    def _augment_waveform(self, wav: torch.Tensor) -> torch.Tensor:
        if not self.augment or not self.aug.get("enabled", False):
            return wav
        max_gain = float(self.aug.get("random_gain_db", 0.0))
        wav = random_gain(wav, max_gain)
        if torch.rand(1).item() < float(self.aug.get("noise_probability", 0.0)):
            wav = add_gaussian_noise(
                wav,
                float(self.aug.get("gaussian_noise_snr_db_min", 5.0)),
                float(self.aug.get("gaussian_noise_snr_db_max", 30.0)),
            )
        return wav.clamp(-1.5, 1.5)

    def _augment_features(self, feats: torch.Tensor) -> torch.Tensor:
        if not self.augment or not self.aug.get("enabled", False):
            return feats
        if torch.rand(1).item() < float(self.aug.get("time_mask_probability", 0.0)):
            feats = time_mask(feats, float(self.aug.get("time_mask_fraction", 0.12)))
        if torch.rand(1).item() < float(self.aug.get("freq_mask_probability", 0.0)):
            feats = freq_mask(feats, int(self.aug.get("freq_mask_bins", 12)))
        return feats

    def __getitem__(self, idx: int) -> Dict:
        seg = self.segments[idx]
        row = self.df.iloc[seg.row_index]
        path = str(row[self.path_column])
        wav = self._load_audio(path, seg.start_sec)
        wav = self._augment_waveform(wav)
        with torch.no_grad():
            feats = self.frontend(wav)
            feats = segment_cmvn(feats)
            feats = self._augment_features(feats)
        target = torch.tensor(float(row[self.target_column]), dtype=torch.float32)
        size_label = int(row.get("size_label", -1)) if "size_label" in row else -1
        return {
            "features": feats.float(),
            "target": target,
            "size_label": torch.tensor(size_label, dtype=torch.long),
            "recording_id": str(row[self.recording_id_column]),
            "audio_path": path,
            "start_sec": torch.tensor(float(seg.start_sec), dtype=torch.float32),
            "duration_sec": torch.tensor(float(seg.duration_sec), dtype=torch.float32),
        }
