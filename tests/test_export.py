from pathlib import Path

import torch

from models.semantic_comm_model import SemanticCommModel


def test_torchscript_export(tmp_path: Path):
    cfg = {
        "in_channels": 1,
        "latent_dim": 64,
        "base_channels": 16,
        "num_classes": 10,
        "enable_recon": True,
        "enable_cls": True,
        "channel": {"type": "awgn", "snr_db": 10.0},
    }
    m = SemanticCommModel(cfg).eval()
    x = torch.randn(1, 1, 28, 28)
    ts = torch.jit.trace(m, x, strict=False)
    p = tmp_path / "m.ts"
    ts.save(str(p))
    loaded = torch.jit.load(str(p))
    y = loaded(x)
    assert "logits" in y
