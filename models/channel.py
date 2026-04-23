from __future__ import annotations

import torch
import torch.nn as nn


class ChannelSimulator(nn.Module):
    """支持 AWGN + dropout(丢包) + 带宽压缩。"""

    def __init__(self, channel_cfg):
        super().__init__()
        self.ctype = channel_cfg.get("type", "awgn")
        self.snr_db = float(channel_cfg.get("snr_db", 15.0))
        self.dropout_p = float(channel_cfg.get("dropout_p", 0.0))
        self.bandwidth_ratio = float(channel_cfg.get("bandwidth_ratio", 1.0))

    def _awgn(self, z):
        signal_power = z.pow(2).mean(dim=1, keepdim=True).clamp(min=1e-8)
        snr = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr
        noise = torch.randn_like(z) * noise_power.sqrt()
        return z + noise

    def _dropout(self, z):
        if self.dropout_p <= 0:
            return z
        mask = (torch.rand_like(z) > self.dropout_p).float()
        return z * mask

    def _bandwidth(self, z):
        if self.bandwidth_ratio >= 1.0:
            return z
        k = max(1, int(z.size(1) * self.bandwidth_ratio))
        out = torch.zeros_like(z)
        out[:, :k] = z[:, :k]
        return out

    def forward(self, z):
        if self.ctype == "awgn":
            return self._awgn(z)
        if self.ctype == "dropout":
            return self._dropout(z)
        if self.ctype == "bandwidth":
            return self._bandwidth(z)
        if self.ctype == "mixed":
            return self._bandwidth(self._dropout(self._awgn(z)))
        return z
