from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.semantic_comm_model import SemanticCommModel
from utils.config import load_config
from utils.io import ensure_dir


def main():
    parser = argparse.ArgumentParser(description="导出 TorchScript")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cpu")
    model = SemanticCommModel(cfg["model"]).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    c = cfg["model"]["in_channels"]
    size = 32 if c == 3 else 28
    example = torch.randn(1, c, size, size)

    scripted = torch.jit.trace(model, example, strict=False)
    out = args.out or str(Path(cfg["deployment"]["export_path"]) / "model.ts")
    ensure_dir(Path(out).parent)
    scripted.save(out)
    print(f"TorchScript saved: {out}")


if __name__ == "__main__":
    main()
