from __future__ import annotations

import argparse
from pathlib import Path

import torch

from datasets import build_dataloaders
from models.semantic_comm_model import SemanticCommModel
from utils.config import load_config
from utils.io import ensure_dir


def main():
    parser = argparse.ArgumentParser(description="导出语义中间特征")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out", default="outputs/features/features.pt")
    parser.add_argument("--max_batches", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    _, val_loader, test_loader = build_dataloaders(cfg)
    loader = val_loader if args.split == "val" else test_loader

    model = SemanticCommModel(cfg["model"]).to(device)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    feats, labels = [], []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= args.max_batches:
                break
            x = x.to(device)
            z = model.encode_to_latent(x)
            feats.append(z.cpu())
            labels.append(y)

    ensure_dir(Path(args.out).parent)
    torch.save({"features": torch.cat(feats), "labels": torch.cat(labels)}, args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
