from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
