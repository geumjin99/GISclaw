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

"""Small HTTP calls from the server itself (tiles, a local model server).

The Python inside the desktop application is not the system's Python and may
not see the system's certificate store, so verification uses certifi's
bundle when it is present — the same roots the OpenAI and Anthropic SDKs
already rely on — with the interpreter's defaults as the fallback.
"""
import json
import ssl
import urllib.request

_ctx = None


def ssl_context() -> ssl.SSLContext:
    global _ctx
    if _ctx is None:
        ctx = ssl.create_default_context()
        try:
            import certifi
            ctx.load_verify_locations(certifi.where())
        except Exception:
            pass
        _ctx = ctx
    return _ctx


def fetch(url: str, timeout: float = 8.0, data: dict = None, headers: dict = None):
    """GET (or POST json) -> (bytes, content_type). Raises on failure."""
    hdrs = {"Content-Type": "application/json"}
    hdrs.update(headers or {})
    req = urllib.request.Request(url, headers=hdrs)
    body = json.dumps(data).encode() if data is not None else None
    with urllib.request.urlopen(req, data=body, timeout=timeout, context=ssl_context()) as r:
        return r.read(), (r.headers.get("Content-Type") or "").split(";")[0].strip()


def fetch_json(url: str, timeout: float = 8.0, data: dict = None):
    raw, _ = fetch(url, timeout=timeout, data=data)
    return json.loads(raw.decode("utf-8") or "null")
