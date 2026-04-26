from __future__ import annotations

from pathlib import Path

import torch
from torch.optim import AdamW

from training.evaluator import evaluate
from training.losses import compute_losses
from utils.io import ensure_dir, save_json


class Trainer:
    def __init__(self, model, cfg, device, logger, out_dir):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.logger = logger
        self.out_dir = Path(out_dir)
        self.optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
        self.history = []
        ensure_dir(self.out_dir)

    def save_ckpt(self, name, epoch, best_val):
        path = self.out_dir / name
        torch.save({"epoch": epoch, "best_val": best_val, "model": self.model.state_dict()}, path)

    def fit(self, train_loader, val_loader):
        best_val = float("inf")
        epochs = self.cfg["training"]["epochs"]

        for epoch in range(1, epochs + 1):
            self.model.train()
            train_loss, n = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                losses = compute_losses(out, y, x, self.cfg)
                loss = losses["total"]

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["training"].get("grad_clip", 1.0))
                self.optimizer.step()

                bs = x.size(0)
                train_loss += loss.item() * bs
                n += bs

            val = evaluate(self.model, val_loader, self.device, self.cfg)
            rec = {
                "epoch": epoch,
                "train_loss": train_loss / max(n, 1),
                "val_loss": val["loss"],
                "val_acc": val["acc"],
                "val_psnr": val["psnr"],
            }
            self.history.append(rec)
            self.logger.info(rec)

            self.save_ckpt("last.pt", epoch, best_val)
            if val["loss"] < best_val:
                best_val = val["loss"]
                self.save_ckpt("best.pt", epoch, best_val)

        save_json({"history": self.history}, self.out_dir / "history.json")
        return self.history
