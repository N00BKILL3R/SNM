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
    parser = argparse.ArgumentParser(description="动态量化")
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

    q_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    out = args.out or str(Path(cfg["deployment"]["quantized_path"]) / "model_dynamic_q.pt")
    ensure_dir(Path(out).parent)
    torch.save(q_model.state_dict(), out)
    print(f"quantized state dict saved: {out}")


if __name__ == "__main__":
    main()
