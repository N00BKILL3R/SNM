from __future__ import annotations

import socket

import torch

from models.partition import Budget, select_partition_point
from models.semantic_comm_model import SemanticCommModel
from services.protocol import recv_packet, send_packet


def run_edge_client(cfg, ckpt="", max_steps=5):
    model = SemanticCommModel(cfg["model"]).eval()
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)

    budget = Budget(
        latency_budget_ms=cfg["partition"]["latency_budget_ms"],
        bandwidth_budget_kb=cfg["partition"]["bandwidth_budget_kb"],
        edge_compute_budget=cfg["partition"]["edge_compute_budget"],
    )
    point = cfg["partition"]["manual_point"] if cfg["partition"]["mode"] == "manual" else select_partition_point(cfg["model"], budget)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect((cfg["service"]["host"], cfg["service"]["cloud_port"]))
        print(f"[edge] connected cloud, partition={point}")
        for _ in range(max_steps):
            c = cfg["model"]["in_channels"]
            s = 32 if c == 3 else 28
            x = torch.randn(1, c, s, s)
            z = model.channel(model.encode_to_latent(x))
            send_packet(sock, {"latent": z, "cmd": "infer"})
            rsp = recv_packet(sock)
            pred = rsp["logits"].argmax(dim=1).item()
            print(f"[edge] pred={pred}, recon_shape={tuple(rsp['recon'].shape)}")
        send_packet(sock, {"cmd": "stop"})
        _ = recv_packet(sock)
