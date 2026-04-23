import torch

from models.semantic_comm_model import SemanticCommModel


def test_model_output_shapes():
    cfg = {
        "in_channels": 3,
        "latent_dim": 128,
        "base_channels": 16,
        "num_classes": 10,
        "enable_recon": True,
        "enable_cls": True,
        "channel": {"type": "awgn", "snr_db": 10.0},
    }
    model = SemanticCommModel(cfg)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    assert y["logits"].shape == (4, 10)
    assert y["recon"].shape == (4, 3, 32, 32)
