from __future__ import annotations

import socket

import torch

from models.semantic_comm_model import SemanticCommModel
from services.protocol import recv_packet, send_packet


def run_cloud_server(cfg, ckpt=""):
    host = cfg["service"]["host"]
    port = cfg["service"]["cloud_port"]
    model = SemanticCommModel(cfg["model"]).eval()
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(1)
        print(f"[cloud] listening at {host}:{port}")
        conn, addr = s.accept()
        with conn:
            print(f"[cloud] connected by {addr}")
            while True:
                pkt = recv_packet(conn)
                if pkt is None:
                    break
                if pkt.get("cmd") == "stop":
                    send_packet(conn, {"ok": True})
                    break
                latent = pkt["latent"]
                with torch.no_grad():
                    logits = model.cls_head(latent)
                    recon = model.decoder(latent)
                send_packet(conn, {"logits": logits.cpu(), "recon": recon.cpu()})
