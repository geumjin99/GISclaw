"""Basemap: settings, the tile proxy and its cache, an MBTiles file, and 'none'."""
import json
import os
import socket
import sqlite3
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

PNG = bytes.fromhex("89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d4944415478da63f8cfc0f01f0005000102a12ee6b40000000049454e44ae426082")


class FakeTiles(BaseHTTPRequestHandler):
    hits = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        FakeTiles.hits.append(self.path)
        if "key=secret" not in self.path:
            self.send_response(401); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(PNG)))
        self.end_headers()
        self.wfile.write(PNG)


@pytest.fixture(scope="module")
def tiles():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = HTTPServer(("127.0.0.1", port), FakeTiles)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{port}/t/{{z}}/{{x}}/{{y}}.png?key={{key}}"
    srv.shutdown()


def test_default_and_providers(client):
    cfg = client.get("/api/settings/basemap").json()
    assert cfg["provider"] == "esri-street" and cfg["tiles"] and cfg["ready"]
    ids = {p["id"] for p in cfg["providers"]}
    assert {"esri-gray", "esri-imagery", "osm", "maptiler", "mapbox", "custom", "mbtiles", "none"} <= ids


def test_custom_provider_key_proxy_and_cache(client, tiles):
    r = client.post("/api/settings/basemap", json={"provider": "custom", "url": tiles,
                                                    "key": "secret", "attribution": "© test"})
    assert r.status_code == 200, r.text
    cfg = r.json()
    assert cfg["ready"] and cfg["masked_key"] == "se…et" and "secret" not in json.dumps(cfg)
    FakeTiles.hits.clear()
    t = client.get("/api/basemap/tile/3/4/2")
    assert t.status_code == 200 and t.headers["content-type"] == "image/png" and t.content == PNG
    assert FakeTiles.hits == ["/t/3/4/2.png?key=secret"]        # the key went upstream, not to the page
    t2 = client.get("/api/basemap/tile/3/4/2")
    assert t2.content == PNG and len(FakeTiles.hits) == 1        # served from the cache
    assert client.get("/api/settings/basemap").json()["cache_bytes"] == len(PNG)
    # a bad key is not the page's problem: a transparent answer
    client.post("/api/settings/basemap", json={"key": "wrong"})
    t3 = client.get("/api/basemap/tile/3/4/3")
    assert t3.status_code == 204 and "401" in t3.headers["x-tile-error"]
    client.post("/api/settings/basemap/clear_cache")
    assert client.get("/api/settings/basemap").json()["cache_bytes"] == 0
    # the check reaches the source and says so, cache or not
    client.post("/api/settings/basemap", json={"key": "secret"})
    chk = client.get("/api/settings/basemap/check").json()
    assert chk["ok"] and "image/png" in chk["detail"]
    client.post("/api/settings/basemap", json={"key": "wrong"})
    chk = client.get("/api/settings/basemap/check").json()
    assert not chk["ok"] and "401" in chk["detail"]
    assert client.get("/api/settings/basemap").json()["cache"] is True


def test_mbtiles_offline_file(client):
    path = os.path.join(os.environ["GISCLAW_WORKSPACE"], "base.mbtiles")
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE metadata (name TEXT, value TEXT)")
    con.execute("INSERT INTO metadata VALUES ('format', 'png')")
    con.execute("CREATE TABLE tiles (zoom_level INTEGER, tile_column INTEGER, tile_row INTEGER, tile_data BLOB)")
    con.execute("INSERT INTO tiles VALUES (2, 1, 3 - 2, ?)", (PNG,))   # xyz (2,1,2) stored TMS-flipped
    con.commit(); con.close()
    cfg = client.post("/api/settings/basemap", json={"provider": "mbtiles", "mbtiles": path}).json()
    assert cfg["ready"], cfg
    assert client.get("/api/basemap/tile/2/1/2").content == PNG
    assert client.get("/api/basemap/tile/2/0/0").status_code == 204
    cfg = client.post("/api/settings/basemap", json={"mbtiles": path + ".missing"}).json()
    assert not cfg["ready"] and "not found" in cfg["problem"]


def test_none_and_out_of_range(client):
    cfg = client.post("/api/settings/basemap", json={"provider": "none"}).json()
    assert cfg["ready"] and not cfg["tiles"]
    assert client.get("/api/basemap/tile/2/1/2").status_code == 204
    client.post("/api/settings/basemap", json={"provider": "osm"})
    assert client.get("/api/basemap/tile/2/9/9").status_code == 204   # x,y beyond 2^z
    client.post("/api/settings/basemap", json={"provider": "esri-street"})
