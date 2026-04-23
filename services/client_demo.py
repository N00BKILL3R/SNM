from __future__ import annotations

import argparse
import threading
import time

from services.cloud_server import run_cloud_server
from services.edge_server import run_edge_client
from utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description="本地双进程 edge-cloud 演示")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", default="")
    args = parser.parse_args()

    cfg = load_config(args.config)

    t = threading.Thread(target=run_cloud_server, args=(cfg, args.ckpt), daemon=True)
    t.start()
    time.sleep(1.0)
    run_edge_client(cfg, args.ckpt, max_steps=3)


if __name__ == "__main__":
    main()
