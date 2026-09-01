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

"""The map behind the data: where its tiles come from.

The page never fetches tiles from a provider itself. It asks this server,
which fetches them, keeps a copy in the data folder, and serves that copy
next time — so a key never appears in the page, a blocked or slow provider
does not blank the map, and an area once viewed is still there offline.
Three kinds of source: a hosted service (with or without a key), a template
of your own, and an MBTiles file on disk, which needs no network at all.
"""
import os
import random
import re
import shutil
import sqlite3
import urllib.error

from app.net import fetch

PROVIDERS = {
    # Esri's classic tile services are served without a key, with attribution.
    "esri-gray": {
        "display": "Esri Light Gray (no key · no buildings, zoom to 16)", "key": False,
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Tiles © Esri — Esri, HERE, Garmin, © OpenStreetMap contributors", "max_zoom": 16},
    "esri-street": {
        "display": "Esri Street Map (no key)", "key": False,
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Tiles © Esri — Esri, HERE, Garmin, © OpenStreetMap contributors", "max_zoom": 19},
    "esri-topo": {
        "display": "Esri Topographic (no key)", "key": False,
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Tiles © Esri — Esri, HERE, Garmin, © OpenStreetMap contributors", "max_zoom": 19},
    "esri-imagery": {
        "display": "Esri World Imagery (no key)", "key": False,
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Tiles © Esri — Esri, Maxar, Earthstar Geographics, and the GIS User Community", "max_zoom": 19},
    "opentopomap": {
        "display": "OpenTopoMap (no key)", "key": False,
        "url": "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", "sub": "abc",
        "attribution": "© OpenStreetMap contributors, SRTM · © OpenTopoMap (CC-BY-SA)", "max_zoom": 17},
    "osm": {
        "display": "OpenStreetMap (no key · light use only)", "key": False,
        "url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        "attribution": "© OpenStreetMap contributors", "max_zoom": 19,
        "hint": "The OpenStreetMap Foundation's own tile servers are for light, occasional use; "
                "for regular work use another source or an MBTiles file."},
    "maptiler": {
        "display": "MapTiler (key)", "key": True, "docs": "https://cloud.maptiler.com/account/keys/",
        "url": "https://api.maptiler.com/maps/streets-v2/{z}/{x}/{y}.png?key={key}",
        "attribution": "© MapTiler © OpenStreetMap contributors", "max_zoom": 20},
    "mapbox": {
        "display": "Mapbox (key)", "key": True, "docs": "https://account.mapbox.com/access-tokens/",
        "url": "https://api.mapbox.com/styles/v1/mapbox/light-v11/tiles/256/{z}/{x}/{y}{r}?access_token={key}",
        "attribution": "© Mapbox © OpenStreetMap contributors", "max_zoom": 22},
    "thunderforest": {
        "display": "Thunderforest (key)", "key": True, "docs": "https://www.thunderforest.com/docs/apikeys/",
        "url": "https://tile.thunderforest.com/atlas/{z}/{x}/{y}.png?apikey={key}",
        "attribution": "© Thunderforest © OpenStreetMap contributors", "max_zoom": 22},
    "custom": {
        "display": "Custom XYZ template", "key": None,
        "url": "", "attribution": "", "max_zoom": 22,
        "hint": "Any {z}/{x}/{y} service — a national portal, a company server, a tileserver of your own, "
                "or a keyed CARTO basemap. Use {key} in the template for a token, {s} for a subdomain, {r} for @2x tiles."},
    "mbtiles": {
        "display": "MBTiles file (offline)", "key": False,
        "url": "", "attribution": "", "max_zoom": 22,
        "hint": "A raster .mbtiles on this computer. QGIS makes one from any layers: "
                "Processing → Raster tools → Generate XYZ tiles (MBTiles)."},
    "none": {"display": "No basemap (data only)", "key": False, "url": "", "attribution": "", "max_zoom": 22},
}

DEFAULTS = {"provider": "esri-street", "key": "", "url": "", "attribution": "",
            "mbtiles": "", "cache": True, "version": 1}

USER_AGENT = "GISclaw (+https://github.com/geumjin99/GISclaw)"
_CONTENT_EXT = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}


def settings(store) -> dict:
    data = store.load().get("basemap") or {}
    out = dict(DEFAULTS)
    out.update({k: v for k, v in data.items() if k in DEFAULTS})
    if out["provider"] not in PROVIDERS:
        out["provider"] = "esri-street"
    return out


def save(store, body: dict) -> dict:
    """Store the choice; an all-mask key means 'keep the one you have'."""
    from app.settings_store import _is_mask
    cur = settings(store)
    new = dict(cur)
    provider = str(body.get("provider") or cur["provider"])
    if provider not in PROVIDERS:
        raise ValueError(f"unknown basemap provider '{provider}'")
    new["provider"] = provider
    key = body.get("key")
    if key is not None and not _is_mask(str(key)):
        new["key"] = str(key).strip()
    if "url" in body:
        new["url"] = str(body.get("url") or "").strip()
    if "attribution" in body:
        new["attribution"] = str(body.get("attribution") or "").strip()
    if "mbtiles" in body:
        new["mbtiles"] = str(body.get("mbtiles") or "").strip()
    if "cache" in body:
        new["cache"] = bool(body["cache"])
    new["version"] = int(cur.get("version", 1)) + 1
    data = store.load()
    data["basemap"] = new
    store.save(data)
    return new


def public(store) -> dict:
    """What the page needs — with the key masked and the template resolved."""
    from app.settings_store import mask_key
    s = settings(store)
    p = PROVIDERS[s["provider"]]
    attribution = s["attribution"] if s["provider"] in ("custom", "mbtiles") else p["attribution"]
    ready = True
    problem = ""
    if p["key"] is True and not s["key"]:
        ready, problem = False, "This provider needs a key."
    elif s["provider"] == "custom" and "{z}" not in s["url"]:
        ready, problem = False, "The template needs {z}, {x} and {y}."
    elif s["provider"] == "mbtiles":
        if not s["mbtiles"]:
            ready, problem = False, "Choose an .mbtiles file."
        elif not os.path.isfile(s["mbtiles"]):
            ready, problem = False, f"File not found: {s['mbtiles']}"
    return {
        "provider": s["provider"], "display": p["display"], "attribution": attribution,
        "max_zoom": p.get("max_zoom", 22), "needs_key": p["key"] is True,
        "masked_key": mask_key(s["key"]), "url": s["url"], "mbtiles": s["mbtiles"],
        "cache": s["cache"], "version": s["version"], "ready": ready, "problem": problem,
        "tiles": s["provider"] != "none",
        "providers": [{"id": k, "display": v["display"], "needs_key": v["key"] is True,
                       "docs": v.get("docs", ""), "hint": v.get("hint", "")} for k, v in PROVIDERS.items()],
    }


# ------------------------------------------------------------------- tiles --
def check(store) -> dict:
    """Fetch one tile from the configured source, bypassing the cache."""
    import time
    s = settings(store)
    if s["provider"] == "none":
        return {"ok": True, "detail": "no basemap"}
    t0 = time.time()
    saved = s["cache"]
    try:
        data = store.load()
        # temporarily uncached so the probe really reaches the source
        data.setdefault("basemap", {})["cache"] = False
        store.save(data)
        res, err = tile(store, 3, 4, 2)
    finally:
        data = store.load()
        data.setdefault("basemap", {})["cache"] = saved
        store.save(data)
    ms = int((time.time() - t0) * 1000)
    if res is None:
        return {"ok": False, "detail": err, "ms": ms}
    return {"ok": True, "detail": f"{res[1]}, {len(res[0])} bytes", "ms": ms}


def cache_dir(store) -> str:
    return os.path.join(store.dir, "tiles")


def cache_size(store) -> int:
    total = 0
    for root, _d, files in os.walk(cache_dir(store)):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def clear_cache(store) -> None:
    shutil.rmtree(cache_dir(store), ignore_errors=True)


def _template(s: dict) -> str:
    p = PROVIDERS[s["provider"]]
    return s["url"] if s["provider"] == "custom" else p["url"]


def _cache_key(s: dict) -> str:
    """One cache folder per distinct source, so switching providers never mixes tiles."""
    src = s["provider"] if s["provider"] != "custom" else re.sub(r"[^a-z0-9]+", "-", s["url"].lower())[:60]
    return src or "custom"


def _from_mbtiles(path: str, z: int, x: int, y: int):
    if not os.path.isfile(path):
        return None, "mbtiles file not found"
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        fmt = "png"
        try:
            row = con.execute("SELECT value FROM metadata WHERE name='format'").fetchone()
            if row and row[0]:
                fmt = str(row[0]).lower()
        except sqlite3.Error:
            pass
        if fmt in ("pbf", "mvt"):
            return None, "vector MBTiles are not supported; export raster tiles"
        tms_y = (1 << z) - 1 - y          # MBTiles count rows from the south
        row = con.execute("SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                          (z, x, tms_y)).fetchone()
    finally:
        con.close()
    if not row:
        return None, "no tile"
    ctype = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "webp": "image/webp"}.get(fmt, "image/png")
    return (bytes(row[0]), ctype), ""


def tile(store, z: int, x: int, y: int, r: str = ""):
    """((bytes, content_type), error). A miss on a 'none' provider is not an error."""
    s = settings(store)
    if s["provider"] == "none":
        return None, "no basemap"
    if not (0 <= z <= 22 and 0 <= x < (1 << z) and 0 <= y < (1 << z)):
        return None, "tile out of range"
    if s["provider"] == "mbtiles":
        return _from_mbtiles(s["mbtiles"], z, x, y)

    r = "@2x" if r == "@2x" and "{r}" in _template(s) else ""
    folder = os.path.join(cache_dir(store), _cache_key(s) + ("-2x" if r else ""), str(z), str(x))
    if s["cache"]:
        for ext in _CONTENT_EXT.values():
            hit = os.path.join(folder, f"{y}.{ext}")
            if os.path.isfile(hit):
                with open(hit, "rb") as f:
                    return (f.read(), next(k for k, v in _CONTENT_EXT.items() if v == ext)), ""

    tpl = _template(s)
    if "{z}" not in tpl:
        return None, "no tile template"
    url = (tpl.replace("{z}", str(z)).replace("{x}", str(x)).replace("{y}", str(y))
              .replace("{r}", r).replace("{key}", s["key"])
              .replace("{s}", random.choice(PROVIDERS[s["provider"]].get("sub") or "a")))
    try:
        data, ctype = fetch(url, timeout=10, headers={"User-Agent": USER_AGENT})
        ctype = ctype or "image/png"
    except urllib.error.HTTPError as e:
        return None, f"provider answered {e.code}"
    except Exception as e:
        return None, f"provider unreachable: {e}"
    if not data or not ctype.startswith("image/"):
        return None, "not an image"
    if s["cache"]:
        ext = _CONTENT_EXT.get(ctype, "png")
        try:
            os.makedirs(folder, exist_ok=True)
            tmp = os.path.join(folder, f".{y}.{ext}.part")
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, os.path.join(folder, f"{y}.{ext}"))
        except OSError:
            pass
    return (data, ctype), ""
