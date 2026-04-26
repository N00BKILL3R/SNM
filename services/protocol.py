from __future__ import annotations

import pickle
import socket
import struct


def send_packet(sock: socket.socket, obj):
    data = pickle.dumps(obj)
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_packet(sock: socket.socket):
    header = _recv_all(sock, 4)
    if not header:
        return None
    length = struct.unpack("!I", header)[0]
    payload = _recv_all(sock, length)
    return pickle.loads(payload)


def _recv_all(sock: socket.socket, n: int):
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data
