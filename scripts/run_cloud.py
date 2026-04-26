from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.cloud_server import run_cloud_server
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="启动云端服务")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default="")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_cloud_server(cfg, args.ckpt)


if __name__ == "__main__":
    main()
