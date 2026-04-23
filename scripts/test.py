from __future__ import annotations

import torch

from datasets import build_dataloaders
from models.semantic_comm_model import SemanticCommModel
from training.evaluator import evaluate
from utils.config import load_config, parse_common_args


def main():
    parser = parse_common_args("测试")
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device if args.device else "cpu")

    _, _, test_loader = build_dataloaders(cfg)
    model = SemanticCommModel(cfg["model"]).to(device)
    if not args.ckpt:
        raise ValueError("test 需要 --ckpt")
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)

    print(evaluate(model, test_loader, device, cfg))


if __name__ == "__main__":
    main()
