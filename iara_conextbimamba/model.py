from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    """Conformer-style local convolution module for time-frequency acoustic patterns."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("ConformerConvModule kernel_size should be odd for same-length padding.")
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        y = x.transpose(1, 2)
        y = self.pointwise_in(y)
        y = F.glu(y, dim=1)
        y = self.depthwise(y)
        y = self.batch_norm(y)
        y = F.silu(y)
        y = self.pointwise_out(y)
        y = self.dropout(y)
        return y.transpose(1, 2)


class PureTorchMamba(nn.Module):
    """A pure-PyTorch selective diagonal SSM fallback.

    This follows the Mamba layer structure closely enough for research prototyping:
    input projection -> depthwise local convolution -> input-conditioned B/C/dt ->
    diagonal selective scan -> gated output projection. For full-speed training on IARA,
    install `mamba-ssm`; the wrapper below will use it automatically.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.d_conv = d_conv

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        self.dropout = nn.Dropout(dropout)

        # Diagonal A with small Gaussian perturbation, matching the useful ablation setting.
        base = torch.arange(1, d_state + 1, dtype=torch.float32).log().repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(base + 0.01 * torch.randn_like(base))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.constant_(self.dt_proj.bias, -3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        batch, length, _ = x.shape
        xz = self.in_proj(x)
        x_part, z_part = xz.chunk(2, dim=-1)

        u = self.conv1d(x_part.transpose(1, 2))[:, :, :length].transpose(1, 2)
        u = F.silu(u)
        z = F.silu(z_part)

        B_t = self.B_proj(u)  # (B, T, N)
        C_t = self.C_proj(u)  # (B, T, N)
        dt = F.softplus(self.dt_proj(u)).clamp_max(1.0)  # (B, T, E)

        A = -torch.exp(self.A_log).to(dtype=u.dtype)  # (E, N), stable negative diagonal

        # Vectorized diagonal recurrence. For h_t = a_t * h_{t-1} + b_t with h_0=0:
        # h_t = prod_{j<=t} a_j * sum_{i<=t} b_i / prod_{j<=i} a_j.
        # This is less numerically robust than the CUDA selective-scan kernel but is fast enough
        # as a research/debug fallback for moderate segment lengths.
        a = torch.exp(dt.unsqueeze(-1) * A.view(1, 1, self.d_inner, self.d_state)).clamp(max=1.0)
        b = dt.unsqueeze(-1) * B_t.unsqueeze(2) * u.unsqueeze(-1)

        if length > 4096:
            state = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=u.dtype)
            outputs = []
            for t in range(length):
                state = a[:, t] * state + b[:, t]
                y_t = (state * C_t[:, t].unsqueeze(1)).sum(dim=-1)
                y_t = y_t + self.D.to(u.dtype).view(1, -1) * u[:, t]
                outputs.append(y_t)
            y = torch.stack(outputs, dim=1)
        else:
            p = torch.cumprod(a.float(), dim=1).clamp_min(1e-12).to(dtype=u.dtype)
            h = p * torch.cumsum(b / p, dim=1)
            y = (h * C_t.unsqueeze(2)).sum(dim=-1) + self.D.to(u.dtype).view(1, 1, -1) * u
        y = y * z
        y = self.out_proj(y)
        return self.dropout(y)


class MambaLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        backend = backend.lower()
        if backend not in {"auto", "mamba_ssm", "torch"}:
            raise ValueError("backend must be one of: auto, mamba_ssm, torch")

        use_mamba_ssm = backend in {"auto", "mamba_ssm"}
        self.backend = "torch"
        if use_mamba_ssm:
            try:
                from mamba_ssm import Mamba  # type: ignore

                self.layer = Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                self.dropout = nn.Dropout(dropout)
                self.backend = "mamba_ssm"
            except Exception as exc:
                if backend == "mamba_ssm":
                    raise RuntimeError("backend='mamba_ssm' requested but mamba_ssm could not be imported") from exc
                self.layer = PureTorchMamba(d_model, d_state, d_conv, expand, dropout)
                self.dropout = nn.Identity()
        else:
            self.layer = PureTorchMamba(d_model, d_state, d_conv, expand, dropout)
            self.dropout = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.layer(x))


class ExtBiMamba(nn.Module):
    """External bidirectional Mamba: separate forward/backward Mamba paths."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        self.forward_mamba = MambaLayer(d_model, d_state, d_conv, expand, dropout=0.0, backend=backend)
        self.backward_mamba = MambaLayer(d_model, d_state, d_conv, expand, dropout=0.0, backend=backend)
        self.dropout = nn.Dropout(dropout)

    @property
    def backend(self) -> str:
        return getattr(self.forward_mamba, "backend", "unknown")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_f = self.forward_mamba(x)
        y_b = torch.flip(self.backward_mamba(torch.flip(x, dims=[1])), dims=[1])
        return self.dropout(y_f + y_b)


class ConExtBiMambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        mamba_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        mamba_cfg = dict(mamba_cfg or {})
        backend = mamba_cfg.get("backend", "auto")
        d_state = int(mamba_cfg.get("d_state", 16))
        d_conv = int(mamba_cfg.get("d_conv", 4))
        expand = int(mamba_cfg.get("expand", 2))

        self.ffn1_norm = nn.LayerNorm(d_model)
        self.ffn1 = FeedForward(d_model, d_ff, dropout)
        self.mamba_norm = nn.LayerNorm(d_model)
        self.mamba = ExtBiMamba(d_model, d_state, d_conv, expand, dropout=dropout, backend=backend)
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = ConformerConvModule(d_model, kernel_size=conv_kernel_size, dropout=dropout)
        self.ffn2_norm = nn.LayerNorm(d_model)
        self.ffn2 = FeedForward(d_model, d_ff, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))
        x = x + self.mamba(self.mamba_norm(x))
        x = x + self.conv(self.conv_norm(x))
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        return self.final_norm(x)


class AttentiveStatsPooling(nn.Module):
    def __init__(self, d_model: int, hidden: Optional[int] = None) -> None:
        super().__init__()
        hidden = hidden or d_model
        self.attn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(x), dim=1)  # (B, T, 1)
        mean = torch.sum(weights * x, dim=1)
        var = torch.sum(weights * (x - mean.unsqueeze(1)).pow(2), dim=1).clamp_min(1e-6)
        std = torch.sqrt(var)
        return torch.cat([mean, std], dim=-1)


class UnderwaterConExtBiMambaDetector(nn.Module):
    """Conformer encoder with MHSA replaced by ExtBiMamba for ship/background detection."""

    def __init__(
        self,
        n_mels: int = 128,
        d_model: int = 192,
        num_layers: int = 6,
        d_ff: int = 768,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        pooling: str = "attentive_stats",
        mamba: Optional[Dict] = None,
        num_binary_outputs: int = 1,
        num_size_classes: int = 0,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model
        self.num_size_classes = int(num_size_classes)

        self.input_norm = nn.LayerNorm(n_mels)
        self.input_proj = nn.Linear(n_mels, d_model)
        self.input_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2, groups=1),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [
                ConExtBiMambaBlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                    mamba_cfg=mamba,
                )
                for _ in range(num_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        pooling = pooling.lower()
        self.pooling_name = pooling
        if pooling == "mean":
            self.pool = None
            pooled_dim = d_model
        elif pooling == "stats":
            self.pool = None
            pooled_dim = 2 * d_model
        elif pooling == "attentive_stats":
            self.pool = AttentiveStatsPooling(d_model)
            pooled_dim = 2 * d_model
        else:
            raise ValueError("pooling must be one of: mean, stats, attentive_stats")

        self.binary_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, pooled_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim // 2, num_binary_outputs),
        )
        if self.num_size_classes > 0:
            self.size_head = nn.Sequential(
                nn.LayerNorm(pooled_dim),
                nn.Linear(pooled_dim, pooled_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(pooled_dim // 2, self.num_size_classes),
            )
        else:
            self.size_head = None

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, T, F)
        if features.ndim != 3:
            raise ValueError(f"Expected features of shape (B,T,F), got {tuple(features.shape)}")
        x = self.input_norm(features)
        x = self.input_proj(x)
        x = self.input_conv(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return self.encoder_norm(x)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling_name == "mean":
            return x.mean(dim=1)
        if self.pooling_name == "stats":
            mean = x.mean(dim=1)
            std = x.std(dim=1).clamp_min(1e-6)
            return torch.cat([mean, std], dim=-1)
        return self.pool(x)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encode(features)
        pooled = self._pool(encoded)
        binary_logits = self.binary_head(pooled).squeeze(-1)
        out = {"logits": binary_logits, "embedding": pooled}
        if self.size_head is not None:
            out["size_logits"] = self.size_head(pooled)
        return out

    def backend_summary(self) -> str:
        if not self.layers:
            return "unknown"
        return self.layers[0].mamba.backend


class ResidualCNNBlock(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.net(x))


class SpectrogramResNetClassifier(nn.Module):
    """Compact residual CNN baseline over log-Mel spectrogram images."""

    def __init__(
        self,
        n_mels: int = 64,
        base_channels: int = 24,
        blocks_per_stage: int = 2,
        dropout: float = 0.1,
        num_binary_outputs: int = 1,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.SiLU(),
        )
        stages = []
        channels = base_channels
        for stage_idx in range(3):
            if stage_idx > 0:
                next_channels = channels * 2
                stages.append(
                    nn.Sequential(
                        nn.Conv2d(channels, next_channels, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(next_channels),
                        nn.SiLU(),
                    )
                )
                channels = next_channels
            for _ in range(blocks_per_stage):
                stages.append(ResidualCNNBlock(channels, dropout=dropout))
        self.encoder = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_binary_outputs),
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        # features: (B, T, F) -> image: (B, 1, F, T)
        x = features.transpose(1, 2).unsqueeze(1)
        x = self.stem(x)
        x = self.encoder(x)
        embedding = self.pool(x).flatten(1)
        logits = self.head(embedding).squeeze(-1)
        return {"logits": logits, "embedding": embedding}

    def backend_summary(self) -> str:
        return "resnet_spectrogram"


class ConformerMHSABlock(nn.Module):
    """Conformer block using multi-head self-attention instead of ExtBiMamba."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ffn1_norm = nn.LayerNorm(d_model)
        self.ffn1 = FeedForward(d_model, d_ff, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv_norm = nn.LayerNorm(d_model)
        self.conv = ConformerConvModule(d_model, kernel_size=conv_kernel_size, dropout=dropout)
        self.ffn2_norm = nn.LayerNorm(d_model)
        self.ffn2 = FeedForward(d_model, d_ff, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))
        y = self.attn_norm(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.attn_dropout(y)
        x = x + self.conv(self.conv_norm(x))
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))
        return self.final_norm(x)


class ConformerDetector(nn.Module):
    """Compact MHSA-based Conformer detector baseline."""

    def __init__(
        self,
        n_mels: int = 64,
        d_model: int = 64,
        num_layers: int = 2,
        d_ff: int = 256,
        num_heads: int = 4,
        conv_kernel_size: int = 15,
        dropout: float = 0.1,
        pooling: str = "attentive_stats",
        num_binary_outputs: int = 1,
    ) -> None:
        super().__init__()
        self.n_mels = n_mels
        self.d_model = d_model
        self.input_norm = nn.LayerNorm(n_mels)
        self.input_proj = nn.Linear(n_mels, d_model)
        self.input_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [
                ConformerMHSABlock(
                    d_model=d_model,
                    d_ff=d_ff,
                    num_heads=num_heads,
                    conv_kernel_size=conv_kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        pooling = pooling.lower()
        self.pooling_name = pooling
        if pooling == "mean":
            self.pool = None
            pooled_dim = d_model
        elif pooling == "stats":
            self.pool = None
            pooled_dim = 2 * d_model
        elif pooling == "attentive_stats":
            self.pool = AttentiveStatsPooling(d_model)
            pooled_dim = 2 * d_model
        else:
            raise ValueError("pooling must be one of: mean, stats, attentive_stats")

        self.binary_head = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, pooled_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(pooled_dim // 2, num_binary_outputs),
        )

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected features of shape (B,T,F), got {tuple(features.shape)}")
        x = self.input_norm(features)
        x = self.input_proj(x)
        x = self.input_conv(x.transpose(1, 2)).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)
        return self.encoder_norm(x)

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling_name == "mean":
            return x.mean(dim=1)
        if self.pooling_name == "stats":
            mean = x.mean(dim=1)
            std = x.std(dim=1).clamp_min(1e-6)
            return torch.cat([mean, std], dim=-1)
        return self.pool(x)

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded = self.encode(features)
        pooled = self._pool(encoded)
        logits = self.binary_head(pooled).squeeze(-1)
        return {"logits": logits, "embedding": pooled}

    def backend_summary(self) -> str:
        return "conformer_mhsa"


def build_model(model_cfg: Dict) -> nn.Module:
    """Build a detector from config while preserving old ConExtBiMamba configs."""

    cfg = dict(model_cfg)
    model_type = str(cfg.pop("type", "conextbimamba")).lower()
    if model_type in {"conextbimamba", "underwater_conextbimamba", "proposed"}:
        return UnderwaterConExtBiMambaDetector(**cfg)
    if model_type in {"resnet", "cnn", "spectrogram_resnet"}:
        allowed = {"n_mels", "base_channels", "blocks_per_stage", "dropout", "num_binary_outputs"}
        return SpectrogramResNetClassifier(**{k: v for k, v in cfg.items() if k in allowed})
    if model_type in {"conformer", "mhsa_conformer"}:
        allowed = {
            "n_mels",
            "d_model",
            "num_layers",
            "d_ff",
            "num_heads",
            "conv_kernel_size",
            "dropout",
            "pooling",
            "num_binary_outputs",
        }
        return ConformerDetector(**{k: v for k, v in cfg.items() if k in allowed})
    raise ValueError(f"Unknown model type: {model_type}")
