from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(Path(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: str) -> Dict[str, Any]:
    default_path = PROJECT_ROOT / "configs" / "default.yaml"
    user_path = Path(config_path)
    if not user_path.is_absolute():
        user_path = PROJECT_ROOT / user_path
    cfg = load_yaml(default_path)
    user_cfg = load_yaml(user_path)
    return deep_update(cfg, user_cfg)


def parse_common_args(desc: str):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--config", type=str, required=True, help="yaml 配置路径")
    parser.add_argument("--ckpt", type=str, default="", help="模型权重路径")
    parser.add_argument("--device", type=str, default="", help="强制设备 cpu/cuda")
    return parser
