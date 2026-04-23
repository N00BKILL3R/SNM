from __future__ import annotations

import torch.nn.functional as F


def compute_losses(outputs, targets, images, cfg):
    losses = {}
    total = 0.0
    if "logits" in outputs:
        cls_loss = F.cross_entropy(outputs["logits"], targets)
        losses["cls_loss"] = cls_loss
        total = total + cfg["training"]["cls_loss_weight"] * cls_loss
    if "recon" in outputs:
        recon_loss = F.mse_loss(outputs["recon"], images)
        losses["recon_loss"] = recon_loss
        total = total + cfg["training"]["recon_loss_weight"] * recon_loss
    losses["total"] = total
    return losses
