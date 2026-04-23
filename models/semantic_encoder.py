from __future__ import annotations

import torch.nn as nn

from models.blocks import ConvBNAct, ResidualBlock


class SemanticEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32, latent_dim=128):
        super().__init__()
        self.stem = ConvBNAct(in_channels, base_channels, s=2)
        self.layer1 = nn.Sequential(ResidualBlock(base_channels), ConvBNAct(base_channels, base_channels * 2, s=2))
        self.layer2 = nn.Sequential(ResidualBlock(base_channels * 2), ConvBNAct(base_channels * 2, base_channels * 4, s=2))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.to_latent = nn.Linear(base_channels * 4, latent_dim)

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        z = self.pool(f2).flatten(1)
        z = self.to_latent(z)
        return {"f1": f1, "f2": f2, "z": z}
