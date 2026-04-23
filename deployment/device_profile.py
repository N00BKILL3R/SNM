from __future__ import annotations

DEVICE_PROFILES = {
    "low_power_edge": {"compute": 0.2, "memory_mb": 1024, "target_latency_ms": 120},
    "normal_edge": {"compute": 0.5, "memory_mb": 4096, "target_latency_ms": 80},
    "cpu_cloud": {"compute": 0.7, "memory_mb": 8192, "target_latency_ms": 60},
    "gpu_cloud": {"compute": 1.0, "memory_mb": 16384, "target_latency_ms": 20},
}


def get_profile(name: str):
    if name not in DEVICE_PROFILES:
        raise KeyError(f"Unknown profile: {name}")
    return DEVICE_PROFILES[name]
