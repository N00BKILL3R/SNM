from __future__ import annotations

from pathlib import Path

import torch

from datasets import build_dataloaders
from models.semantic_comm_model import SemanticCommModel
from training.trainer import Trainer
from utils.config import load_config, parse_common_args
from utils.logger import create_logger
from utils.seed import set_seed
from utils.visualization import save_history_plots


def main():
    parser = parse_common_args("训练")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() and cfg.get("device") != "cpu" else "cpu"))

    out_dir = Path(cfg["project"]["output_root"]) / cfg["project"]["experiment"]
    logger = create_logger("train", out_dir / "train.log")
    logger.info(f"device={device}")

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = SemanticCommModel(cfg["model"]).to(device)

    trainer = Trainer(model, cfg, device, logger, out_dir)
    history = trainer.fit(train_loader, val_loader)
    save_history_plots(history, str(out_dir))


if __name__ == "__main__":
    main()
