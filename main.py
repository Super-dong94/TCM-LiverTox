from __future__ import annotations

import threading
import webbrowser
import os
import socket

import uvicorn

from backend.app import app


HOST = "127.0.0.1"
PORT_CANDIDATES = [7860, 8765, 18000, 5000, 49152, 55000]


def find_available_port() -> int:
    env_port = os.getenv("TOXHERB_PORT")
    candidates = [int(env_port)] + PORT_CANDIDATES if env_port else PORT_CANDIDATES
    for port in candidates:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
            except OSError:
                continue
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def open_browser(url: str) -> None:
    webbrowser.open(url)


if __name__ == "__main__":
    port = find_available_port()
    URL = f"http://{HOST}:{port}"
    print(f"[ToxHERB] Server URL: {URL}")
    threading.Timer(1.5, open_browser, args=(URL,)).start()
    uvicorn.run(app, host=HOST, port=port, reload=False)
