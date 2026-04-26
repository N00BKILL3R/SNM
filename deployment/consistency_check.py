from __future__ import annotations

import argparse
import copy
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
    parser.add_argument("--ts", default="", help="可选：外部 TorchScript 路径；为空时使用当前模型临时trace")
    parser.add_argument("--seed", type=int, default=42, help="随机种子，保证一致性检查可复现")
    parser.add_argument("--disable_channel", action="store_true", help="一致性检查时禁用信道随机扰动（推荐）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = copy.deepcopy(cfg["model"])
    if args.disable_channel:
        model_cfg["channel"] = {
            "type": "none",
            "snr_db": 120.0,
            "dropout_p": 0.0,
            "bandwidth_ratio": 1.0,
        }

    torch.manual_seed(args.seed)
    c = cfg["model"]["in_channels"]
    s = 32 if c == 3 else 28
    x = torch.randn(8, c, s, s)

    base = SemanticCommModel(model_cfg).eval()
    state = torch.load(args.ckpt, map_location="cpu")
    base.load_state_dict(state["model"] if "model" in state else state)

    with torch.no_grad():
        y_base = base(x)

    if args.ts:
        scripted = torch.jit.load(args.ts, map_location="cpu").eval()
    else:
        scripted = torch.jit.trace(base, x, strict=False).eval()
    with torch.no_grad():
        y_ts = scripted(x)

    q_model = torch.quantization.quantize_dynamic(base, {torch.nn.Linear}, dtype=torch.qint8).eval()
    with torch.no_grad():
        y_q = q_model(x)

    threshold = cfg["deployment"].get("consistency_threshold", 0.03)
    report = {
        "diff_base_vs_ts_logits": tensor_diff(y_base["logits"], y_ts["logits"]),
        "diff_base_vs_q_logits": tensor_diff(y_base["logits"], y_q["logits"]),
        "threshold": threshold,
        "pass_base_vs_ts": tensor_diff(y_base["logits"], y_ts["logits"]) <= threshold,
        "pass_base_vs_q": tensor_diff(y_base["logits"], y_q["logits"]) <= threshold,
        "channel_disabled": args.disable_channel,
        "ts_source": args.ts if args.ts else "traced_in_memory",
    }
    print(report)


if __name__ == "__main__":
    main()
