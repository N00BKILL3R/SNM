from __future__ import annotations

import torch

from training.losses import compute_losses
from training.metrics import accuracy, psnr


def evaluate(model, loader, device, cfg):
    model.eval()
    loss_sum, acc_sum, psnr_sum, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            losses = compute_losses(out, y, x, cfg)
            bs = x.size(0)
            loss_sum += losses["total"].item() * bs
            if "logits" in out:
                acc_sum += accuracy(out["logits"], y) * bs
            if "recon" in out:
                psnr_sum += psnr(out["recon"], x) * bs
            n += bs
    return {
        "loss": loss_sum / max(n, 1),
        "acc": acc_sum / max(n, 1),
        "psnr": psnr_sum / max(n, 1),
    }
