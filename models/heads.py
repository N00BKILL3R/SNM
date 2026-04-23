from __future__ import annotations

import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, num_classes),
        )

    def forward(self, z):
        return self.net(z)
