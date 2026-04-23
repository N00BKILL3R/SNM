from __future__ import annotations

import argparse

from services.edge_server import run_edge_client
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="启动边缘端")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_edge_client(cfg, args.ckpt, args.steps)


if __name__ == "__main__":
    main()
