from __future__ import annotations

import torch
import torch.nn as nn

from models.channel import ChannelSimulator
from models.heads import ClassificationHead
from models.semantic_decoder import SemanticDecoder
from models.semantic_encoder import SemanticEncoder


class SemanticCommModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = SemanticEncoder(cfg["in_channels"], cfg["base_channels"], cfg["latent_dim"])
        self.channel = ChannelSimulator(cfg.get("channel", {}))
        self.decoder = SemanticDecoder(cfg["in_channels"], cfg["base_channels"], cfg["latent_dim"], out_size=32 if cfg["in_channels"] == 3 else 28)
        self.cls_head = ClassificationHead(cfg["latent_dim"], cfg["num_classes"])
        self.enable_recon = cfg.get("enable_recon", True)
        self.enable_cls = cfg.get("enable_cls", True)

    def encode_to_latent(self, x):
        return self.encoder(x)["z"]

    def forward(self, x):
        feat = self.encoder(x)
        z = feat["z"]
        z_noisy = self.channel(z)
        out = {"latent": z_noisy}
        if self.enable_recon:
            out["recon"] = self.decoder(z_noisy)
        if self.enable_cls:
            out["logits"] = self.cls_head(z_noisy)
        return out

    def forward_from_partition(self, x=None, packet=None, point="p1"):
        if point == "p0":
            return self.forward(x), None
        if point == "p1":
            z = self.encode_to_latent(x)
            return None, {"type": "latent", "value": z}
        if point == "p2":
            z = packet["value"] if packet else self.channel(self.encode_to_latent(x))
            return {"latent": z, "recon": self.decoder(z), "logits": self.cls_head(z)}, None
        raise ValueError(f"Unsupported partition point: {point}")
