from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import streamlit as st
import torch
import torchaudio
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

PROJECT_IMPORT_ERROR: Exception | None = None
try:
    from iara_conextbimamba.features import (  # noqa: E402
        LogMelExtractor,
        MultiResolutionAcousticExtractor,
        segment_cmvn,
    )
    from iara_conextbimamba.model import build_model  # noqa: E402
except Exception as exc:  # pragma: no cover - exercised only in incomplete release packages.
    PROJECT_IMPORT_ERROR = exc
    LogMelExtractor = None  # type: ignore[assignment]
    MultiResolutionAcousticExtractor = None  # type: ignore[assignment]
    segment_cmvn = None  # type: ignore[assignment]
    build_model = None  # type: ignore[assignment]

APP_VERSION = "1.0.0-windows-deploy"
MAX_PUBLIC_DURATION_SEC = 30.0 * 60.0
LARGE_FILE_MB_WARNING = 500.0
EVENT_PROBABILITY_THRESHOLD = 0.5
BENCHMARK_SAMPLE_RATE = 16000
BENCHMARK_BAG_SECONDS = 121.0
BENCHMARK_BAG_SAMPLES = 1_936_000
FROZEN_SEGMENT_SECONDS = 20.0
FROZEN_SEGMENT_SAMPLES = 320_000
FROZEN_SEGMENT_STARTS = (162_666, 808_000, 1_453_333)
FROZEN_CHECKPOINT = (
    "revision_training/task3_matched_models/frozen_checkpoints/"
    "matched_conextbimamba/seed_20260825/best_checkpoint.pt"
)
FROZEN_DATA_CONFIG: dict[str, Any] = {
    "sample_rate": BENCHMARK_SAMPLE_RATE,
    "segment_seconds": FROZEN_SEGMENT_SECONDS,
    "feature_mode": "multi_resolution_logmel_cqt",
    "n_fft": 2048,
    "hop_length": 512,
    "win_length": 2048,
    "n_mels": 128,
    "f_min": 10.0,
    "f_max": 8000.0,
    "multi_resolution": {
        "mel_resolutions": [
            {
                "n_fft": 1024,
                "win_length": 1024,
                "hop_length": 512,
                "n_mels": 64,
                "f_min": 10.0,
                "f_max": 8000.0,
            }
        ],
        "cqt": {
            "enabled": True,
            "n_fft": 4096,
            "win_length": 4096,
            "hop_length": 512,
            "n_bins": 84,
            "f_min": 20.0,
            "f_max": 8000.0,
        },
    },
}


@dataclass(frozen=True)
class ModeSpec:
    name: str
    role: str
    checkpoint: str
    aggregation: str
    threshold: float
    score_label: str
    batch_size: int


MODES: dict[str, ModeSpec] = {
    "ConExtBiMamba-Default": ModeSpec(
        name="ConExtBiMamba-Default",
        role="Frozen primary offline research mode",
        checkpoint=FROZEN_CHECKPOINT,
        aggregation="mean",
        threshold=0.2287546408673127,
        score_label="mean of three fixed segment probabilities",
        batch_size=3,
    ),
    "ConExtBiMamba-Conservative": ModeSpec(
        name="ConExtBiMamba-Conservative",
        role="Frozen secondary offline research mode; not proven operationally superior",
        checkpoint=FROZEN_CHECKPOINT,
        aggregation="mean_log_survival",
        threshold=0.6510023052576646,
        score_label="mean-log-survival score from three fixed segments",
        batch_size=3,
    ),
}

PUBLIC_MODES = [
    "ConExtBiMamba-Default",
    "ConExtBiMamba-Conservative",
]

ADVANCED_MODES: list[str] = []

AUDIO_TYPES = ["wav", "flac", "aiff", "aif", "ogg", "mp3"]


def apply_page_style() -> None:
    st.markdown(
        """
        <style>
        :root {
            --detector-teal: #0f766e;
            --detector-green: #16a34a;
            --detector-purple: #7c3aed;
            --detector-ink: #172033;
            --detector-soft: #eef6f5;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
            max-width: 1280px;
        }
        .decision-card {
            border-left: 6px solid var(--detector-green);
            background: linear-gradient(90deg, #ecfdf5 0%, #ffffff 72%);
            padding: 1rem 1.15rem;
            margin: 0.25rem 0 1rem 0;
            border-radius: 8px;
            border-top: 1px solid #d7ebe6;
            border-right: 1px solid #d7ebe6;
            border-bottom: 1px solid #d7ebe6;
        }
        .decision-card.background {
            border-left-color: #0f766e;
            background: linear-gradient(90deg, #eff6ff 0%, #ffffff 72%);
        }
        .decision-card.review {
            border-left-color: #d97706;
            background: linear-gradient(90deg, #fffbeb 0%, #ffffff 72%);
        }
        .decision-title {
            font-size: 1.45rem;
            line-height: 1.2;
            margin: 0;
            font-weight: 750;
            color: var(--detector-ink);
        }
        .decision-meta {
            margin-top: 0.35rem;
            color: #475569;
            font-size: 0.92rem;
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.65rem 0.8rem;
            color: var(--detector-ink);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        div[data-testid="stMetric"] [data-testid="stMetricValue"],
        div[data-testid="stMetric"] [data-testid="stMetricDelta"],
        div[data-testid="stMetric"] p {
            color: var(--detector-ink) !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stMetric"] {
            background: #111827;
            border-color: #374151;
            color: #f8fafc;
        }
        section[data-testid="stSidebar"] div[data-testid="stMetric"] label,
        section[data-testid="stSidebar"] div[data-testid="stMetric"] [data-testid="stMetricLabel"],
        section[data-testid="stSidebar"] div[data-testid="stMetric"] [data-testid="stMetricValue"],
        section[data-testid="stSidebar"] div[data-testid="stMetric"] [data-testid="stMetricDelta"],
        section[data-testid="stSidebar"] div[data-testid="stMetric"] p {
            color: #f8fafc !important;
        }
        .small-label {
            color: #64748b;
            font-size: 0.84rem;
            margin-top: -0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def dataframe_table(data: Any, **kwargs: Any) -> None:
    try:
        st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(data, use_container_width=True, **kwargs)


def validate_mode_resources(mode_name: str) -> bool:
    if PROJECT_IMPORT_ERROR is not None:
        st.error("The ConExtBiMamba project modules could not be imported.")
        st.info(
            "Keep app.py with the iara_conextbimamba/ source folder, trained weights, and run/config files, "
            "then relaunch the dashboard."
        )
        st.code(str(PROJECT_IMPORT_ERROR))
        return False

    required_files = [
        PROJECT_ROOT / "iara_conextbimamba" / "features.py",
        PROJECT_ROOT / "iara_conextbimamba" / "model.py",
    ]
    missing = [path for path in required_files if not path.exists()]
    if missing:
        st.error("Required project source files are missing.")
        for path in missing:
            st.code(str(path))
        return False

    spec = MODES[mode_name]
    checkpoint = PROJECT_ROOT / spec.checkpoint
    if not checkpoint.exists():
        st.error(f"Checkpoint not found: {checkpoint}")
        st.info("Download the trained weights or update the checkpoint path in app.py.")
        return False
    return True


@st.cache_data(show_spinner=False)
def sha256_file(path: str) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return "missing"
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_hash(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def frontend_description(data_cfg: dict[str, Any]) -> str:
    feature_mode = str(data_cfg.get("feature_mode", "logmel"))
    n_mels = int(data_cfg.get("n_mels", 64))
    sample_rate = int(data_cfg.get("sample_rate", 16000))
    segment_seconds = float(data_cfg.get("segment_seconds", 10.0))
    if "multi" in feature_mode.lower():
        return (
            f"{feature_mode}; 128-bin Mel (FFT 2048), 64-bin Mel (FFT 1024), "
            f"84-bin log-spaced/CQT-style branch (FFT 4096), hop 512; "
            f"{sample_rate} Hz; {segment_seconds:g} s segments; segment CMVN"
        )
    return f"{feature_mode}; {n_mels} Mel bins; {sample_rate} Hz; {segment_seconds:g} s segments; segment CMVN"


def inspect_audio_file(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    meta: dict[str, Any] = {
        "file_size_mb": file_path.stat().st_size / (1024.0 * 1024.0) if file_path.exists() else 0.0,
    }
    try:
        info = sf.info(str(file_path))
        sample_rate = float(info.samplerate)
        frames = float(info.frames)
        meta.update(
            {
                "sample_rate": sample_rate,
                "frames": frames,
                "duration_sec": frames / max(sample_rate, 1.0),
                "channels": int(info.channels),
                "format": str(info.format),
            }
        )
    except Exception as exc:
        meta["metadata_error"] = str(exc)
    return meta


def show_audio_warnings(meta: dict[str, Any]) -> None:
    if meta.get("metadata_error"):
        st.info("Audio duration could not be inspected before inference. The detector will still try to decode the file.")
    duration = float(meta.get("duration_sec", 0.0) or 0.0)
    size_mb = float(meta.get("file_size_mb", 0.0) or 0.0)
    if 0.0 < duration < BENCHMARK_BAG_SECONDS:
        st.warning(
            "This file is shorter than 121 seconds. It can be inspected, but its score and decision are explicitly "
            "nonbenchmark diagnostics because the frozen three-position evidence bag cannot be reproduced."
        )
    elif duration >= BENCHMARK_BAG_SECONDS:
        st.info(
            "The detector will evaluate the first 121 seconds using the three frozen 20-second segment starts. "
            "Audio beyond 121 seconds is not used in this recording-level decision."
        )
    if duration > MAX_PUBLIC_DURATION_SEC:
        st.warning("This recording is longer than 30 minutes. For large files, offline batch processing is recommended.")
    if size_mb > LARGE_FILE_MB_WARNING:
        st.warning("This audio file is larger than 500 MB. Browser uploads may be slow or memory-heavy.")


def uncertainty_flag(score: float, threshold: float, margin: float = 0.05) -> str:
    if threshold <= 1.0:
        return "review" if abs(score - threshold) <= margin else "confident"
    return "review" if abs(score - threshold) <= 0.10 * threshold else "confident"


def event_summaries(segments: pd.DataFrame, threshold: float = EVENT_PROBABILITY_THRESHOLD) -> pd.DataFrame:
    columns = [
        "event_index",
        "start_sec",
        "end_sec",
        "duration_sec",
        "n_segments",
        "max_probability",
        "mean_probability",
    ]
    if segments.empty:
        return pd.DataFrame(columns=columns)

    events: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def flush() -> None:
        if not current:
            return
        start = min(float(row["start_sec"]) for row in current)
        end = max(float(row["end_sec"]) for row in current)
        probs = [float(row["ship_probability"]) for row in current]
        events.append(
            {
                "event_index": len(events) + 1,
                "start_sec": start,
                "end_sec": end,
                "duration_sec": max(0.0, end - start),
                "n_segments": len(current),
                "max_probability": max(probs),
                "mean_probability": float(np.mean(probs)),
            }
        )
        current.clear()

    for row in segments.to_dict("records"):
        if float(row["ship_probability"]) >= threshold:
            current.append(row)
        else:
            flush()
    flush()
    return pd.DataFrame(events, columns=columns)


def segments_with_report_metadata(result: dict[str, Any]) -> pd.DataFrame:
    segments = result["segments"].copy()
    summary = result["summary"]
    for key in [
        "file",
        "mode",
        "aggregation",
        "score_label",
        "score",
        "threshold",
        "prediction",
        "uncertainty_flag",
        "checkpoint_sha256",
        "config_hash",
        "aggregation_config_hash",
        "segment_seconds",
        "benchmark_equivalent",
        "evaluation_scope",
        "source_duration_sec",
        "evaluated_duration_sec",
        "waveform_gain_normalization",
        "segment_normalization",
        "device",
        "torch_version",
    ]:
        segments[key] = summary.get(key)
    return segments


def report_payload(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": result["summary"],
        "events": result["events"].to_dict("records"),
        "segments": result["segments"].to_dict("records"),
    }


def build_frontend(data_cfg: dict[str, Any]) -> torch.nn.Module:
    if LogMelExtractor is None or MultiResolutionAcousticExtractor is None:
        raise RuntimeError("Feature extractors are unavailable because the project modules did not import correctly.")
    feature_mode = str(data_cfg.get("feature_mode", "logmel")).lower()
    sample_rate = int(data_cfg.get("sample_rate", 16000))
    if feature_mode in {"multi_resolution_logmel_cqt", "multi_resolution", "logmel_cqt"}:
        return MultiResolutionAcousticExtractor(
            sample_rate=sample_rate,
            n_fft=int(data_cfg.get("n_fft", 2048)),
            hop_length=int(data_cfg.get("hop_length", 512)),
            win_length=int(data_cfg.get("win_length", data_cfg.get("n_fft", 2048))),
            n_mels=int(data_cfg.get("n_mels", 128)),
            f_min=float(data_cfg.get("f_min", 10.0)),
            f_max=float(data_cfg.get("f_max", sample_rate / 2)),
            config=data_cfg.get("multi_resolution", {}),
        )
    return LogMelExtractor(
        sample_rate=sample_rate,
        n_fft=int(data_cfg.get("n_fft", 1024)),
        hop_length=int(data_cfg.get("hop_length", 512)),
        win_length=int(data_cfg.get("win_length", data_cfg.get("n_fft", 1024))),
        n_mels=int(data_cfg.get("n_mels", 64)),
        f_min=float(data_cfg.get("f_min", 10.0)),
        f_max=float(data_cfg.get("f_max", sample_rate / 2)),
    )


@st.cache_resource(show_spinner=False)
def load_detector(mode_name: str):
    if build_model is None:
        raise RuntimeError("Model builder is unavailable because the project modules did not import correctly.")
    spec = MODES[mode_name]
    checkpoint = PROJECT_ROOT / spec.checkpoint
    # This repository-authored checkpoint contains numpy scalar metadata and is therefore
    # loaded with weights_only=False. Its SHA-256 is exported with every result.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = {"model": ckpt["model_config"], "data": FROZEN_DATA_CONFIG}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"])
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    frontend = build_frontend(cfg["data"]).to(device).eval()
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model, frontend, cfg, device, trainable_params


def read_audio(path: str | Path, target_sr: int) -> torch.Tensor:
    path = str(path)
    try:
        wav, sr = torchaudio.load(path)
    except Exception:
        data, sr = sf.read(path, always_2d=True)
        wav = torch.from_numpy(data.T.astype(np.float32))
    wav = wav.mean(dim=0).float()
    if int(sr) != int(target_sr):
        wav = torchaudio.functional.resample(wav, orig_freq=int(sr), new_freq=int(target_sr))
    # Preserve physical amplitude relationships: the frozen evaluation used no gain
    # normalization or waveform-level centering. Segment CMVN remains in the frontend.
    return wav.contiguous()


def segment_plan(num_samples: int, sample_rate: int) -> tuple[list[int], bool, str]:
    """Return the frozen K=3 plan or an explicitly nonbenchmark diagnostic plan."""
    if sample_rate != BENCHMARK_SAMPLE_RATE:
        raise ValueError(f"Segment planning requires {BENCHMARK_SAMPLE_RATE} Hz audio after resampling")
    if num_samples >= BENCHMARK_BAG_SAMPLES:
        return list(FROZEN_SEGMENT_STARTS), True, "frozen_121s_three_segment_contract"
    max_start = max(0, num_samples - FROZEN_SEGMENT_SAMPLES)
    if max_start == 0:
        starts = [0]
    else:
        starts = sorted(set(int(round(x)) for x in np.linspace(0, max_start, num=3)))
    return starts, False, "nonbenchmark_short_file_diagnostic"


def aggregate_score(probabilities: np.ndarray, aggregation: str) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.size == 0:
        return 0.0
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("Segment probabilities must be in [0, 1]")
    if aggregation == "mean":
        return float(probs.mean())
    if aggregation == "mean_log_survival":
        return float(1.0 - np.exp(np.log(1.0 - probs + 1e-8).mean()))
    raise ValueError(f"Unsupported frozen aggregation: {aggregation}")


@torch.no_grad()
def run_detection(audio_path: str | Path, mode_name: str) -> dict[str, Any]:
    spec = MODES[mode_name]
    model, frontend, cfg, device, trainable_params = load_detector(mode_name)
    data_cfg = cfg["data"]
    sample_rate = int(data_cfg.get("sample_rate", 16000))
    segment_seconds = float(data_cfg.get("segment_seconds", 10.0))
    wav = read_audio(audio_path, sample_rate)
    source_num_samples = int(wav.numel())
    starts, benchmark_equivalent, evaluation_scope = segment_plan(source_num_samples, sample_rate)
    if benchmark_equivalent:
        wav = wav[:BENCHMARK_BAG_SAMPLES]
    seg_len = int(round(segment_seconds * sample_rate))

    rows: list[dict[str, Any]] = []
    batch_feats: list[torch.Tensor] = []
    batch_meta: list[tuple[int, float]] = []

    def flush_batch() -> None:
        if not batch_feats:
            return
        features = torch.stack(batch_feats).to(device)
        logits = model(features)["logits"]
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(float)
        for (start, duration), prob in zip(batch_meta, probs):
            rows.append(
                {
                    "start_sec": start / sample_rate,
                    "duration_sec": duration,
                    "ship_probability": float(prob),
                    "segment_prediction_at_0_5": int(prob >= 0.5),
                }
            )
        batch_feats.clear()
        batch_meta.clear()

    frontend.eval()
    for start in starts:
        seg = wav[start : start + seg_len]
        duration = min(segment_seconds, max(0.0, (wav.numel() - start) / sample_rate))
        if seg.numel() < seg_len:
            seg = torch.nn.functional.pad(seg, (0, seg_len - seg.numel()))
        feats = frontend(seg.to(device))
        if segment_cmvn is None:
            raise RuntimeError("segment_cmvn is unavailable because the project modules did not import correctly.")
        feats = segment_cmvn(feats).float().cpu()
        batch_feats.append(feats)
        batch_meta.append((start, duration))
        if len(batch_feats) >= spec.batch_size:
            flush_batch()
    flush_batch()

    segments = pd.DataFrame(rows).sort_values("start_sec").reset_index(drop=True)
    if not segments.empty:
        segments["end_sec"] = segments["start_sec"] + segments["duration_sec"]
    score = aggregate_score(segments["ship_probability"].to_numpy(), spec.aggregation)
    prediction = "ship" if score >= spec.threshold else "background"
    review_state = uncertainty_flag(score, spec.threshold)
    events = event_summaries(segments)
    top_segment = segments.loc[segments["ship_probability"].idxmax()].to_dict() if not segments.empty else {}
    checkpoint_path = PROJECT_ROOT / spec.checkpoint
    summary = {
        "app_version": APP_VERSION,
        "file": Path(audio_path).name,
        "mode": spec.name,
        "role": spec.role,
        "aggregation": spec.aggregation,
        "score_label": spec.score_label,
        "score": score,
        "threshold": spec.threshold,
        "prediction": prediction,
        "uncertainty_flag": review_state,
        "n_segments": int(len(segments)),
        "n_events": int(len(events)),
        "sample_rate": sample_rate,
        "segment_seconds": segment_seconds,
        "duration_sec": float(source_num_samples / sample_rate),
        "source_duration_sec": float(source_num_samples / sample_rate),
        "evaluated_duration_sec": float(wav.numel() / sample_rate),
        "benchmark_equivalent": benchmark_equivalent,
        "evaluation_scope": evaluation_scope,
        "fixed_segment_start_samples": list(FROZEN_SEGMENT_STARTS) if benchmark_equivalent else starts,
        "fixed_segment_start_seconds": [start / sample_rate for start in starts],
        "waveform_gain_normalization": "none",
        "segment_normalization": "per-segment CMVN after frozen frontend",
        "top_segment_start_sec": float(top_segment.get("start_sec", 0.0)),
        "top_segment_probability": float(top_segment.get("ship_probability", 0.0)),
        "checkpoint_path": spec.checkpoint,
        "checkpoint_sha256": sha256_file(str(checkpoint_path)),
        "config_hash": config_hash(cfg),
        "aggregation_config_hash": config_hash(
            {
                "aggregation": spec.aggregation,
                "threshold": spec.threshold,
                "required_segment_starts": FROZEN_SEGMENT_STARTS,
                "bag_samples": BENCHMARK_BAG_SAMPLES,
            }
        ),
        "frontend_description": frontend_description(data_cfg),
        "trainable_parameters": trainable_params,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    return {
        "summary": summary,
        "segments": segments,
        "events": events,
        "waveform": wav.cpu().numpy(),
        "sample_rate": sample_rate,
    }


def save_uploaded_file(uploaded_file) -> Path:
    upload_dir = PROJECT_ROOT / "runs" / "gui_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    data = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower() or ".wav"
    digest = hashlib.sha1(data).hexdigest()[:12]
    safe_name = "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in Path(uploaded_file.name).stem)
    path = upload_dir / f"{safe_name}_{digest}{suffix}"
    path.write_bytes(data)
    return path


@st.cache_data(show_spinner=False)
def example_files() -> list[str]:
    preferred = ["D-1155.wav", "H-1802.wav", "E-1505.wav", "D-1036.wav", "A-0420.wav"]
    search_dirs = [
        PROJECT_ROOT / "examples" / "wavs",
        PROJECT_ROOT / "data" / "iara_full",
    ]
    for data_dir in search_dirs:
        if not data_dir.exists():
            continue
        wavs = sorted(data_dir.rglob("*.wav"))
        if not wavs:
            continue
        selected = []
        for name in preferred:
            stem = name.removesuffix(".wav")
            matches = [p for p in wavs if p.name == name or stem in p.stem]
            selected.extend(matches[:1])
        return [str(p) for p in (selected or wavs[:8])]
    return []


def format_seconds(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    minutes, seconds = divmod(float(value), 60.0)
    return f"{int(minutes):02d}:{seconds:05.2f}"


def waveform_figure(wav: np.ndarray, sample_rate: int) -> plt.Figure:
    max_points = 25000
    step = max(1, int(math.ceil(len(wav) / max_points)))
    y = wav[::step]
    t = np.arange(len(y)) * step / sample_rate
    fig, ax = plt.subplots(figsize=(10.5, 2.6))
    ax.plot(t, y, color="#0f766e", lw=0.65)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def spectrogram_figure(wav: np.ndarray, sample_rate: int) -> plt.Figure:
    nperseg = 2048 if sample_rate >= 16000 else 1024
    noverlap = nperseg // 2
    freqs, times, spec = signal.spectrogram(
        wav,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="magnitude",
    )
    keep = freqs <= min(8000, sample_rate / 2)
    db = 20 * np.log10(np.maximum(spec[keep], 1e-8))
    vmax = np.percentile(db, 99.5)
    vmin = vmax - 80
    fig, ax = plt.subplots(figsize=(10.5, 3.25))
    pcm = ax.pcolormesh(times, freqs[keep] / 1000.0, db, shading="auto", cmap="magma", vmin=vmin, vmax=vmax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (kHz)")
    ax.set_title("Spectrogram")
    fig.colorbar(pcm, ax=ax, label="dB", pad=0.01)
    fig.tight_layout()
    return fig


def timeline_figure(segments: pd.DataFrame, spec: ModeSpec) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10.5, 3.0))
    ax.plot(segments["start_sec"], segments["ship_probability"], marker="o", ms=4, lw=1.4, color="#2563eb")
    if spec.threshold <= 1.0:
        ax.axhline(spec.threshold, color="#dc2626", ls="--", lw=1.0, label="recording threshold")
    else:
        ax.text(
            0.01,
            0.94,
            f"Aggregate evidence threshold = {spec.threshold:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="#7c3aed",
            fontsize=8.5,
        )
    ax.set_xlabel("Segment start time (s)")
    ax.set_ylabel("p(ship)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Segment Probability Timeline")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def combined_png(result: dict[str, Any], mode_name: str) -> bytes:
    spec = MODES[mode_name]
    wav = result["waveform"]
    sr = result["sample_rate"]
    segments = result["segments"]
    summary = result["summary"]

    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 1.15, 1.0], hspace=0.38)

    ax1 = fig.add_subplot(gs[0])
    step = max(1, int(math.ceil(len(wav) / 25000)))
    y = wav[::step]
    t = np.arange(len(y)) * step / sr
    ax1.plot(t, y, color="#0f766e", lw=0.55)
    ax1.set_ylabel("Amplitude")
    ax1.set_title("Waveform")
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(alpha=0.22)

    ax2 = fig.add_subplot(gs[1])
    nperseg = 2048 if sr >= 16000 else 1024
    freqs, times, specgram = signal.spectrogram(wav, fs=sr, window="hann", nperseg=nperseg, noverlap=nperseg // 2, scaling="density", mode="magnitude")
    keep = freqs <= min(8000, sr / 2)
    db = 20 * np.log10(np.maximum(specgram[keep], 1e-8))
    vmax = np.percentile(db, 99.5)
    pcm = ax2.pcolormesh(times, freqs[keep] / 1000.0, db, shading="auto", cmap="magma", vmin=vmax - 80, vmax=vmax)
    ax2.set_ylabel("kHz")
    ax2.set_title("Spectrogram")
    fig.colorbar(pcm, ax=ax2, label="dB", pad=0.01)

    ax3 = fig.add_subplot(gs[2])
    ax3.plot(segments["start_sec"], segments["ship_probability"], marker="o", ms=3, lw=1.2, color="#2563eb")
    if spec.threshold <= 1.0:
        ax3.axhline(spec.threshold, color="#dc2626", ls="--", lw=1.0)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("p(ship)")
    ax3.set_ylim(-0.03, 1.03)
    ax3.set_title("Segment Probability Timeline")
    ax3.grid(alpha=0.22)

    fig.suptitle(
        f"{summary['file']} | {summary['mode']} | {summary['prediction'].upper()} | score={summary['score']:.4g}",
        x=0.01,
        y=0.995,
        ha="left",
        fontsize=13,
        weight="bold",
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def load_benchmark_table() -> pd.DataFrame:
    path = PROJECT_ROOT / "revision_manuscript/task7_final_revision/tables/Main_Table_2_Corrected_Matched_Benchmark.csv"
    if not path.exists():
        return pd.DataFrame()
    source = pd.read_csv(path)
    columns = {
        "False-positive rate": "False alarm rate",
        "Balanced accuracy": "Balanced accuracy",
        "Miss rate": "Miss rate",
        "F1": "F1",
        "AUROC": "AUROC",
        "AUPRC": "AUPRC",
    }
    result = pd.DataFrame({"Model": source["Model"], "Role": "Corrected matched benchmark; five-seed mean"})
    for source_col, target_col in columns.items():
        result[target_col] = source[source_col].astype(str).str.split().str[0].astype(float)
    return result


def benchmark_figure(df: pd.DataFrame) -> plt.Figure:
    plot = df.dropna(subset=["Balanced accuracy"]).copy()
    colors = ["#64748b", "#2563eb", "#d97706", "#0891b2", "#16a34a", "#7c3aed"][: len(plot)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(plot["Model"], plot["Balanced accuracy"], color=colors, edgecolor="#1e293b", linewidth=0.5)
    axes[0].set_ylim(0.8, 1.0)
    axes[0].set_ylabel("Balanced accuracy")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_title("Held-out IARA test split")
    rates = plot.dropna(subset=["False alarm rate", "Miss rate"])
    x = np.arange(len(rates))
    axes[1].bar(x - 0.18, rates["False alarm rate"], width=0.36, color="#ef4444", label="FAR")
    axes[1].bar(x + 0.18, rates["Miss rate"], width=0.36, color="#f59e0b", label="Miss")
    axes[1].set_xticks(x, rates["Model"], rotation=25, ha="right")
    axes[1].set_ylim(0.0, 0.25)
    axes[1].set_title("Error trade-off")
    axes[1].legend()
    fig.tight_layout()
    return fig


def decision_card(summary: dict[str, Any]) -> None:
    is_ship = summary["prediction"] == "ship"
    review = summary.get("uncertainty_flag") == "review"
    card_class = " review" if review else ("" if is_ship else " background")
    title = "Review recommended" if review else ("Ship detected" if is_ship else "Background")
    decision_text = summary["prediction"]
    st.markdown(
        f"""
        <div class="decision-card{card_class}">
            <p class="decision-title">{title}</p>
            <div class="decision-meta">
                {summary["mode"]} | {summary["score_label"]}: {summary["score"]:.4g} |
                threshold: {summary["threshold"]:.4g} | decision: {decision_text} |
                uncertainty: {summary.get("uncertainty_flag", "n/a")} | segments: {summary["n_segments"]} |
                scope: {summary.get("evaluation_scope", "unknown")}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_single_analysis(mode_name: str) -> None:
    source = st.radio("Audio source", ["Upload file", "Bundled example"], horizontal=True)
    audio_path: Path | None = None
    uploaded = None
    if source == "Upload file":
        uploaded = st.file_uploader("Single-file WAV upload", type=AUDIO_TYPES)
        if uploaded is not None:
            audio_path = save_uploaded_file(uploaded)
            st.audio(uploaded)
    else:
        examples = example_files()
        if not examples:
            st.warning("No bundled example recordings were found. Please upload a WAV file.")
            return
        labels = [Path(p).name for p in examples]
        selected = st.selectbox("Example recording", labels)
        audio_path = Path(examples[labels.index(selected)])
        st.audio(str(audio_path))

    if audio_path is not None:
        audio_meta = inspect_audio_file(audio_path)
        show_audio_warnings(audio_meta)

    run_clicked = st.button("Run detection", type="primary", disabled=audio_path is None)
    if run_clicked and audio_path is not None:
        with st.spinner("Running detector..."):
            st.session_state["single_result"] = run_detection(audio_path, mode_name)
            st.session_state["single_mode"] = mode_name

    result = st.session_state.get("single_result")
    if result is None:
        st.info("Select an audio file to run recording-level ship/background detection.")
        return
    if st.session_state.get("single_mode") != mode_name:
        st.warning("The displayed result was produced with a different mode. Run detection again for the selected mode.")

    summary = result["summary"]
    if not summary.get("benchmark_equivalent", False):
        st.error(
            "NONBENCHMARK DIAGNOSTIC: this file does not reproduce the frozen 121-second, three-position evidence bag. "
            "Do not compare this decision with manuscript performance estimates."
        )
    decision_card(summary)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Score", f"{summary['score']:.4g}")
    m2.metric("Threshold", f"{summary['threshold']:.4g}")
    m3.metric("Duration", format_seconds(summary["duration_sec"]))
    m4.metric("Top segment", f"{summary['top_segment_probability']:.3f}")
    with st.expander("Inference Metadata"):
        st.json(
            {
                "app_version": summary["app_version"],
                "checkpoint_path": summary["checkpoint_path"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "config_hash": summary["config_hash"],
                "aggregation_config_hash": summary["aggregation_config_hash"],
                "frontend_description": summary["frontend_description"],
                "benchmark_equivalent": summary["benchmark_equivalent"],
                "evaluation_scope": summary["evaluation_scope"],
                "fixed_segment_start_samples": summary["fixed_segment_start_samples"],
                "waveform_gain_normalization": summary["waveform_gain_normalization"],
                "segment_normalization": summary["segment_normalization"],
                "device": summary["device"],
                "torch_version": summary["torch_version"],
                "trainable_parameters": summary["trainable_parameters"],
            }
        )

    waveform_tab, spectrogram_tab, timeline_tab, table_tab, events_tab = st.tabs(
        ["Waveform", "Spectrogram", "Timeline", "Segments", "Events"]
    )
    with waveform_tab:
        fig = waveform_figure(result["waveform"], result["sample_rate"])
        st.pyplot(fig, clear_figure=True)
    with spectrogram_tab:
        fig = spectrogram_figure(result["waveform"], result["sample_rate"])
        st.pyplot(fig, clear_figure=True)
    with timeline_tab:
        fig = timeline_figure(result["segments"], MODES[mode_name])
        st.pyplot(fig, clear_figure=True)
    with table_tab:
        dataframe_table(result["segments"], hide_index=True)
    with events_tab:
        if result["events"].empty:
            st.info("No ship-positive segment events were found at the 0.5 segment threshold.")
        else:
            dataframe_table(result["events"], hide_index=True)

    csv_bytes = segments_with_report_metadata(result).to_csv(index=False).encode("utf-8")
    json_bytes = json.dumps(report_payload(result), indent=2).encode("utf-8")
    png_bytes = combined_png(result, mode_name)
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download CSV", csv_bytes, file_name=f"{Path(summary['file']).stem}_segments.csv", mime="text/csv")
    c2.download_button("Download JSON", json_bytes, file_name=f"{Path(summary['file']).stem}_decision.json", mime="application/json")
    c3.download_button("Download PNG", png_bytes, file_name=f"{Path(summary['file']).stem}_analysis.png", mime="image/png")


def render_batch(mode_name: str) -> None:
    files = st.file_uploader("Batch processing", type=AUDIO_TYPES, accept_multiple_files=True)
    if not files:
        st.info("Upload multiple audio files to run the selected detector mode as a batch.")
        return
    run_clicked = st.button("Run batch", type="primary")
    if run_clicked:
        progress = st.progress(0.0)
        summaries: list[dict[str, Any]] = []
        per_file_segments: dict[str, pd.DataFrame] = {}
        per_file_events: dict[str, pd.DataFrame] = {}
        for idx, uploaded in enumerate(files, start=1):
            try:
                path = save_uploaded_file(uploaded)
                meta = inspect_audio_file(path)
                warnings: list[str] = []
                if float(meta.get("duration_sec", 0.0) or 0.0) > MAX_PUBLIC_DURATION_SEC:
                    warnings.append("longer than 30 minutes")
                if float(meta.get("file_size_mb", 0.0) or 0.0) > LARGE_FILE_MB_WARNING:
                    warnings.append("larger than 500 MB")
                if meta.get("metadata_error"):
                    warnings.append("metadata inspection failed")
                with st.spinner(f"Running {uploaded.name}..."):
                    result = run_detection(path, mode_name)
                summary = result["summary"] | {"status": "ok", "error": "", "file_warning": "; ".join(warnings)}
                summaries.append(summary)
                per_file_segments[uploaded.name] = segments_with_report_metadata(result)
                per_file_events[uploaded.name] = result["events"]
            except Exception as exc:
                summaries.append(
                    {
                        "file": uploaded.name,
                        "mode": mode_name,
                        "status": "failed",
                        "error": str(exc),
                    }
                )
            finally:
                progress.progress(idx / len(files))
        st.session_state["batch_summaries"] = summaries
        st.session_state["batch_segments"] = per_file_segments
        st.session_state["batch_events"] = per_file_events
        st.session_state["batch_mode"] = mode_name

    summaries = st.session_state.get("batch_summaries")
    if not summaries:
        return
    if st.session_state.get("batch_mode") != mode_name:
        st.warning("The displayed batch was produced with a different mode. Run the batch again for the selected mode.")

    summary_df = pd.DataFrame(summaries)
    ok_count = int((summary_df.get("status", pd.Series(dtype=str)) == "ok").sum()) if "status" in summary_df else len(summary_df)
    failed_count = int((summary_df.get("status", pd.Series(dtype=str)) == "failed").sum()) if "status" in summary_df else 0
    review_count = int((summary_df.get("uncertainty_flag", pd.Series(dtype=str)) == "review").sum()) if "uncertainty_flag" in summary_df else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Completed", ok_count)
    c2.metric("Failed", failed_count)
    c3.metric("Review recommended", review_count)
    desired_cols = [
        "file",
        "mode",
        "status",
        "prediction",
        "uncertainty_flag",
        "score_label",
        "score",
        "threshold",
        "n_segments",
        "n_events",
        "duration_sec",
        "top_segment_probability",
        "file_warning",
        "error",
    ]
    cols = [col for col in desired_cols if col in summary_df.columns]
    dataframe_table(summary_df[cols], hide_index=True)

    csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
    json_bytes = json.dumps(summaries, indent=2).encode("utf-8")
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("batch_summary.csv", summary_df.to_csv(index=False))
        zf.writestr("batch_summary.json", json.dumps(summaries, indent=2))
        for name, seg_df in st.session_state.get("batch_segments", {}).items():
            zf.writestr(f"segments/{Path(name).stem}_segments.csv", seg_df.to_csv(index=False))
        for name, events_df in st.session_state.get("batch_events", {}).items():
            zf.writestr(f"events/{Path(name).stem}_events.csv", events_df.to_csv(index=False))
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download Batch CSV", csv_bytes, file_name="batch_summary.csv", mime="text/csv")
    c2.download_button("Download Batch JSON", json_bytes, file_name="batch_summary.json", mime="application/json")
    c3.download_button("Download Batch ZIP", zip_buf.getvalue(), file_name="batch_detection_outputs.zip", mime="application/zip")


def render_benchmark() -> None:
    df = load_benchmark_table()
    if df.empty:
        st.info(
            "The corrected matched-benchmark summary is not present in this installation. "
            "Audio inference remains available, but the comparison panel cannot be rendered."
        )
        return
    dataframe_table(
        df.style.format(
            {
                "Balanced accuracy": "{:.3f}",
                "F1": "{:.3f}",
                "False alarm rate": "{:.3f}",
                "Miss rate": "{:.3f}",
                "AUROC": "{:.3f}",
                "AUPRC": "{:.3f}",
            },
            na_rep="not reported",
        ),
        hide_index=True,
    )
    fig = benchmark_figure(df)
    st.pyplot(fig, clear_figure=True)


def render_about() -> None:
    st.warning(
        "Offline research decision-support prototype only. It is not a real-time detector, safety system, vessel census, "
        "or substitute for qualified human review."
    )
    st.subheader("Model")
    st.write(
        "The dashboard wraps the frozen seed-20260825 ConExtBiMamba checkpoint from the corrected IARA experiment. "
        "Default uses the arithmetic mean of exactly three fixed segment probabilities. Conservative uses the "
        "validation-frozen mean-log-survival mode; it is an alternative operating mode, not an operational guarantee."
    )
    st.write(
        "Architecturally, the detector uses a Conformer-style acoustic encoder in which multi-head self-attention is replaced by "
        "external bidirectional Mamba, while retaining feed-forward, convolutional, residual, and normalization components."
    )
    st.subheader("Modes")
    dataframe_table(
        pd.DataFrame(
            [
                {
                    "Mode": m.name,
                    "Role": m.role,
                    "Aggregation": m.aggregation,
                    "Threshold": m.threshold,
                    "Checkpoint": m.checkpoint,
                }
                for m in MODES.values()
            ]
        ),
        hide_index=True,
    )
    st.subheader("Version And Reproducibility")
    st.write(
        f"GUI version: {APP_VERSION}. Exported CSV and JSON reports include the model mode, aggregation method, threshold, "
        "checkpoint path, checkpoint SHA-256, configuration hash, frontend settings, selected device, and PyTorch version."
    )
    st.subheader("Input And Output")
    st.write(
        "WAV, FLAC, AIFF, OGG, and MP3 inputs are accepted when the local audio stack supports the codec. "
        "Audio is converted to mono and resampled to 16 kHz without gain normalization. Benchmark-contract evaluation uses "
        "a 121-second bag and exactly three 20-second segments starting at samples 162666, 808000, and 1453333. "
        "Shorter files remain inspectable but are prominently labeled nonbenchmark diagnostics."
    )
    st.caption(
        "Uploaded files are processed locally and cached under runs/gui_uploads in this installation. "
        "Delete that folder when retained local copies are no longer needed."
    )


def selected_mode_from_sidebar() -> str:
    with st.sidebar:
        st.header("Detector Mode")
        mode_options = PUBLIC_MODES
        mode_name = st.selectbox("Model mode", mode_options, index=0)
        spec = MODES[mode_name]
        st.caption(spec.role)
        st.metric("Aggregation", spec.aggregation.replace("_", " "))
        st.metric("Threshold", f"{spec.threshold:.4g}")
        st.caption(f"Checkpoint: {spec.checkpoint}")
    return mode_name


def main() -> None:
    st.set_page_config(page_title="Underwater ConExtBiMamba Detector", page_icon="~", layout="wide")
    apply_page_style()
    st.title("Underwater ConExtBiMamba Detector")
    mode_name = selected_mode_from_sidebar()
    if not validate_mode_resources(mode_name):
        st.stop()

    analyze_tab, batch_tab, benchmark_tab, about_tab = st.tabs(
        ["Analyze WAV", "Batch Processing", "Benchmark Comparison", "About"]
    )
    with analyze_tab:
        render_single_analysis(mode_name)
    with batch_tab:
        render_batch(mode_name)
    with benchmark_tab:
        render_benchmark()
    with about_tab:
        render_about()


if __name__ == "__main__":
    main()
