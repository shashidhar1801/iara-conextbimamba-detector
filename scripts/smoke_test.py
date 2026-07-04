#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

import torch

torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from iara_conextbimamba.model import UnderwaterConExtBiMambaDetector
from iara_conextbimamba.utils import count_parameters


def main():
    # Tiny CPU test. The real IARA config should use mamba-ssm on a GPU.
    model = UnderwaterConExtBiMambaDetector(
        n_mels=16,
        d_model=8,
        num_layers=1,
        d_ff=16,
        conv_kernel_size=7,
        dropout=0.1,
        mamba={"backend": "torch", "d_state": 4, "d_conv": 4, "expand": 1},
    )
    x = torch.randn(1, 16, 16)  # [batch, time, n_mels]
    out = model(x)
    assert out["logits"].shape == (1,), out["logits"].shape
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], torch.tensor([1.0]))
    loss.backward()
    print("Smoke test OK")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Backend: {model.backend_summary()}")
    print(f"Logits: {out['logits'].detach().tolist()}")


if __name__ == "__main__":
    main()
