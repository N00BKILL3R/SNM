from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Budget:
    latency_budget_ms: float
    bandwidth_budget_kb: float
    edge_compute_budget: float


def estimate_packet_kb(latent_dim: int, batch_size: int = 1, dtype_bytes: int = 4) -> float:
    return latent_dim * batch_size * dtype_bytes / 1024


def select_partition_point(model_cfg, budget: Budget) -> str:
    packet_kb = estimate_packet_kb(model_cfg["latent_dim"])
    if budget.edge_compute_budget < 0.25:
        return "p0"  # 全部云端
    if packet_kb > budget.bandwidth_budget_kb:
        return "p0"
    if budget.latency_budget_ms < 40 and budget.edge_compute_budget >= 0.5:
        return "p2"  # 更多边缘计算，减少请求轮转
    return "p1"  # 默认边缘编码、云端解码
