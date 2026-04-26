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


def tensor_diff(a, b):
    return (a - b).abs().mean().item()


def main():
    parser = argparse.ArgumentParser(description="一致性检查")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--ts", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)
    c = cfg["model"]["in_channels"]
    s = 32 if c == 3 else 28
    x = torch.randn(8, c, s, s)

    base = SemanticCommModel(cfg["model"]).eval()
    state = torch.load(args.ckpt, map_location="cpu")
    base.load_state_dict(state["model"] if "model" in state else state)

    with torch.no_grad():
        y_base = base(x)

    ts_path = args.ts if args.ts else cfg["deployment"]["export_path"] + "/model.ts"
    scripted = torch.jit.load(ts_path, map_location="cpu").eval()
    with torch.no_grad():
        y_ts = scripted(x)

    q_model = torch.quantization.quantize_dynamic(base, {torch.nn.Linear}, dtype=torch.qint8).eval()
    with torch.no_grad():
        y_q = q_model(x)

    report = {
        "diff_base_vs_ts_logits": tensor_diff(y_base["logits"], y_ts["logits"]),
        "diff_base_vs_q_logits": tensor_diff(y_base["logits"], y_q["logits"]),
        "threshold": cfg["deployment"].get("consistency_threshold", 0.03),
    }
    print(report)


if __name__ == "__main__":
    main()
