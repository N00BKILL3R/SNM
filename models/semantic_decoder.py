from __future__ import annotations

import torch
import torch.nn as nn

from models.blocks import ConvBNAct


class SemanticDecoder(nn.Module):
    def __init__(self, out_channels=3, base_channels=32, latent_dim=128, out_size=32):
        super().__init__()
        self.base_channels = base_channels
        self.out_size = out_size
        self.fc = nn.Linear(latent_dim, base_channels * 4 * 4 * 4)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBNAct(base_channels * 4, base_channels * 2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBNAct(base_channels * 2, base_channels),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_channels, out_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(z.size(0), self.base_channels * 4, 4, 4)
        recon = self.up(x)
        if recon.shape[-1] != self.out_size:
            recon = torch.nn.functional.interpolate(recon, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return recon
