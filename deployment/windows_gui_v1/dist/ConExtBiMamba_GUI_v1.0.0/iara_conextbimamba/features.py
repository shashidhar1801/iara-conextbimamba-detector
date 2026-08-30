from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torchaudio


class LogMelExtractor(torch.nn.Module):
    """Log-mel frontend for underwater passive-acoustic segments.

    Input shape:  (num_samples,) or (batch, num_samples)
    Output shape: (time, n_mels) or (batch, time, n_mels)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: int = 1024,
        n_mels: int = 128,
        f_min: float = 10.0,
        f_max: Optional[float] = None,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.eps = eps
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max if f_max is not None else sample_rate / 2,
            n_mels=n_mels,
            power=2.0,
            normalized=False,
            center=True,
            pad_mode="reflect",
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        single = waveform.ndim == 1
        if single:
            waveform = waveform.unsqueeze(0)
        mel = self.mel(waveform).clamp_min(self.eps)
        logmel = self.to_db(mel)
        logmel = logmel.transpose(-1, -2).contiguous()
        if single:
            logmel = logmel.squeeze(0)
        return logmel


def _power_to_db(power: torch.Tensor, top_db: float = 80.0, eps: float = 1e-10) -> torch.Tensor:
    power_db = 10.0 * torch.log10(power.clamp_min(eps))
    max_db = power_db.amax(dim=(-2, -1), keepdim=True)
    return torch.maximum(power_db, max_db - top_db)


def _log_frequency_filterbank(
    n_freqs: int,
    sample_rate: int,
    n_bins: int,
    f_min: float,
    f_max: float,
) -> torch.Tensor:
    freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs)
    f_min = max(float(f_min), 1.0)
    f_max = min(float(f_max), sample_rate / 2.0)
    edges = torch.logspace(math.log10(f_min), math.log10(f_max), n_bins + 2)
    fb = torch.zeros(n_freqs, n_bins)
    for idx in range(n_bins):
        lower = edges[idx]
        center = edges[idx + 1]
        upper = edges[idx + 2]
        up = (freqs - lower) / (center - lower).clamp_min(1e-6)
        down = (upper - freqs) / (upper - center).clamp_min(1e-6)
        weights = torch.minimum(up, down).clamp_min(0.0)
        if float(weights.sum()) <= 0.0:
            nearest = int(torch.argmin((freqs - center).abs()).item())
            weights[nearest] = 1.0
        fb[:, idx] = weights / weights.sum().clamp_min(1e-6)
    return fb


class LogFrequencyExtractor(torch.nn.Module):
    """CQT-style log-frequency power frontend built from an STFT filterbank."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 4096,
        hop_length: int = 512,
        win_length: int = 4096,
        n_bins: int = 84,
        f_min: float = 20.0,
        f_max: Optional[float] = None,
        eps: float = 1e-10,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=2.0,
            center=True,
            pad_mode="reflect",
        )
        fb = _log_frequency_filterbank(
            n_freqs=n_fft // 2 + 1,
            sample_rate=sample_rate,
            n_bins=n_bins,
            f_min=f_min,
            f_max=f_max if f_max is not None else sample_rate / 2,
        )
        self.register_buffer("filterbank", fb)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        single = waveform.ndim == 1
        if single:
            waveform = waveform.unsqueeze(0)
        spec = self.spec(waveform).clamp_min(self.eps)
        logfreq = torch.einsum("bft,fc->btc", spec, self.filterbank.to(spec.dtype))
        logfreq = _power_to_db(logfreq, eps=self.eps)
        if single:
            logfreq = logfreq.squeeze(0)
        return logfreq.contiguous()


def _align_time(features: torch.Tensor, target_len: int) -> torch.Tensor:
    if features.shape[-2] == target_len:
        return features
    single = features.ndim == 2
    x = features.unsqueeze(0) if single else features
    x = x.transpose(1, 2)
    x = torch.nn.functional.interpolate(x, size=target_len, mode="linear", align_corners=False)
    x = x.transpose(1, 2).contiguous()
    return x.squeeze(0) if single else x


class MultiResolutionAcousticExtractor(torch.nn.Module):
    """Concatenate primary log-Mel, extra log-Mel resolutions, and log-frequency bins."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int = 2048,
        n_mels: int = 128,
        f_min: float = 10.0,
        f_max: Optional[float] = None,
        config: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        config = dict(config or {})
        self.primary = LogMelExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
        )
        mel_resolutions = config.get("mel_resolutions", [])
        self.extra_mels = torch.nn.ModuleList(
            [
                LogMelExtractor(
                    sample_rate=sample_rate,
                    n_fft=int(mel.get("n_fft", n_fft)),
                    hop_length=int(mel.get("hop_length", hop_length)),
                    win_length=int(mel.get("win_length", mel.get("n_fft", n_fft))),
                    n_mels=int(mel.get("n_mels", 64)),
                    f_min=float(mel.get("f_min", f_min)),
                    f_max=float(mel.get("f_max", f_max if f_max is not None else sample_rate / 2)),
                )
                for mel in mel_resolutions
            ]
        )
        cqt_cfg = dict(config.get("cqt", {}))
        self.cqt = None
        if cqt_cfg.get("enabled", False):
            self.cqt = LogFrequencyExtractor(
                sample_rate=sample_rate,
                n_fft=int(cqt_cfg.get("n_fft", 4096)),
                hop_length=int(cqt_cfg.get("hop_length", hop_length)),
                win_length=int(cqt_cfg.get("win_length", cqt_cfg.get("n_fft", 4096))),
                n_bins=int(cqt_cfg.get("n_bins", 84)),
                f_min=float(cqt_cfg.get("f_min", 20.0)),
                f_max=float(cqt_cfg.get("f_max", f_max if f_max is not None else sample_rate / 2)),
            )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        primary = self.primary(waveform)
        target_len = primary.shape[-2]
        features: List[torch.Tensor] = [primary]
        for extractor in self.extra_mels:
            features.append(_align_time(extractor(waveform), target_len))
        if self.cqt is not None:
            features.append(_align_time(self.cqt(waveform), target_len))
        return torch.cat(features, dim=-1).contiguous()


def segment_cmvn(features: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-segment cepstral mean/variance normalization over time and frequency."""
    mean = features.mean(dim=(-2, -1), keepdim=True)
    std = features.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return (features - mean) / std


def random_gain(waveform: torch.Tensor, max_gain_db: float) -> torch.Tensor:
    if max_gain_db <= 0:
        return waveform
    gain_db = torch.empty(1).uniform_(-max_gain_db, max_gain_db).item()
    return waveform * (10.0 ** (gain_db / 20.0))


def add_gaussian_noise(waveform: torch.Tensor, snr_db_min: float, snr_db_max: float) -> torch.Tensor:
    snr_db = torch.empty(1).uniform_(snr_db_min, snr_db_max).item()
    signal_power = waveform.pow(2).mean().clamp_min(1e-12)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = torch.randn_like(waveform) * noise_power.sqrt()
    return waveform + noise


def time_mask(features: torch.Tensor, max_fraction: float = 0.12) -> torch.Tensor:
    if max_fraction <= 0:
        return features
    t = features.shape[-2]
    max_width = max(1, int(math.ceil(t * max_fraction)))
    width = int(torch.randint(1, max_width + 1, (1,)).item())
    start = int(torch.randint(0, max(1, t - width + 1), (1,)).item())
    out = features.clone()
    out[..., start : start + width, :] = 0.0
    return out


def freq_mask(features: torch.Tensor, max_bins: int = 12) -> torch.Tensor:
    if max_bins <= 0:
        return features
    f = features.shape[-1]
    width = int(torch.randint(1, min(max_bins, f) + 1, (1,)).item())
    start = int(torch.randint(0, max(1, f - width + 1), (1,)).item())
    out = features.clone()
    out[..., :, start : start + width] = 0.0
    return out
