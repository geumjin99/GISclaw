#!/usr/bin/env python3
# GISclaw — an LLM agent for geospatial analysis.
# Copyright (C) 2026 Han Jinzhen
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This file is part of GISclaw. GISclaw is free software: you can redistribute
# it and/or modify it under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version. It is distributed in the hope
# that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Affero General Public License in the LICENSE file, or
# <https://www.gnu.org/licenses/>, for more details.

"""Start GISclaw as a desktop application.

Runs the server on a loopback port and opens it in a native window (a
WebView2 window on Windows, a WKWebView window on macOS). Your projects,
settings and keys live in your user data folder, not in the program folder,
so replacing the program never touches them.

    python desktop/launcher.py            # window
    python desktop/launcher.py --browser  # your default browser instead
    python desktop/launcher.py --serve    # server only, no window (prints the URL)

Environment: GISCLAW_DATA overrides the data folder; GISCLAW_PORT pins the port.
"""
import argparse
import os
import platform
import socket
import sys
import threading
import time
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

APP_NAME = "GISclaw"


def data_dir() -> str:
    """Where the user's work goes: the platform's application-data folder."""
    override = os.environ.get("GISCLAW_DATA")
    if override:
        return os.path.abspath(os.path.expanduser(override))
    home = os.path.expanduser("~")
    system = platform.system()
    if system == "Darwin":
        return os.path.join(home, "Library", "Application Support", APP_NAME)
    if system == "Windows":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
        return os.path.join(base, APP_NAME)
    return os.path.join(os.environ.get("XDG_DATA_HOME") or os.path.join(home, ".local", "share"), APP_NAME)


def cjk_font_stack() -> str:
    """Fonts with Chinese, Japanese and Korean glyphs, whichever the OS ships."""
    system = platform.system()
    if system == "Darwin":
        names = ["PingFang SC", "Apple SD Gothic Neo", "Hiragino Sans", "Arial Unicode MS", "Helvetica"]
    elif system == "Windows":
        names = ["Microsoft YaHei", "Malgun Gothic", "Yu Gothic", "Segoe UI", "Arial"]
    else:
        names = ["Noto Sans CJK SC", "Noto Sans CJK KR", "Noto Sans CJK JP", "DejaVu Sans"]
    return ", ".join(names + ["sans-serif"])


def prepare_environment(data: str) -> None:
    os.makedirs(data, exist_ok=True)
    os.environ["GISCLAW_WORKSPACE"] = data
    os.environ.setdefault("GISCLAW_LOG", os.path.join(data, "server.log"))
    # Figures the agent labels in Chinese or Korean need a font that has those
    # glyphs; matplotlib's default does not. Pick from what this OS has.
    mpl_dir = os.path.join(data, ".mplconfig")
    os.makedirs(mpl_dir, exist_ok=True)
    rc = os.path.join(mpl_dir, "matplotlibrc")
    if not os.path.exists(rc):
        with open(rc, "w", encoding="utf-8") as f:
            f.write(f"font.sans-serif: {cjk_font_stack()}\naxes.unicode_minus: False\n")
    os.environ.setdefault("MPLCONFIGDIR", mpl_dir)


def free_port(preferred: int = 8765) -> int:
    for port in (preferred, 0):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("no free port")


def start_server(port: int):
    import uvicorn
    from app.server import app

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="gisclaw-server")
    thread.start()
    for _ in range(600):
        if server.started:
            return server
        if not thread.is_alive():
            raise RuntimeError("the server stopped before it was ready — see server.log")
        time.sleep(0.05)
    raise RuntimeError("the server did not start in time")


def open_window(url: str) -> bool:
    """A native window. False when the platform has no web view to offer."""
    try:
        import webview
    except ImportError:
        return False
    try:
        webview.create_window(APP_NAME, url, width=1440, height=900, min_size=(960, 640))
        webview.start()
        return True
    except Exception as e:                     # no GTK/Qt on this Linux, etc.
        print(f"native window unavailable ({e}); opening the browser instead")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Start GISclaw")
    parser.add_argument("--browser", action="store_true", help="open in the default browser")
    parser.add_argument("--serve", action="store_true", help="server only; print the URL")
    parser.add_argument("--port", type=int, default=int(os.environ.get("GISCLAW_PORT") or 0))
    parser.add_argument("--data", default="", help="data folder (default: the user data folder)")
    args = parser.parse_args()

    data = os.path.abspath(os.path.expanduser(args.data)) if args.data else data_dir()
    prepare_environment(data)
    port = args.port or free_port()
    server = start_server(port)
    url = f"http://127.0.0.1:{port}/"
    print(f"{APP_NAME} is running at {url}\ndata folder: {data}", flush=True)

    try:
        if args.serve:
            while True:
                time.sleep(3600)
        elif args.browser or not open_window(url):
            webbrowser.open(url)
            print("Close this window (Ctrl+C) to stop GISclaw.", flush=True)
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        pass
    finally:
        server.should_exit = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
