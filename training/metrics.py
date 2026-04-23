from __future__ import annotations

import math

import torch


def accuracy(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def topk_accuracy(logits, y, k=3):
    topk = logits.topk(k, dim=1).indices
    hit = (topk == y.unsqueeze(1)).any(dim=1).float().mean().item()
    return hit


def mse(x, y):
    return torch.mean((x - y) ** 2).item()


def psnr(x, y, max_val=1.0):
    m = mse(x, y)
    if m <= 1e-12:
        return 99.0
    return 10.0 * math.log10(max_val * max_val / m)
