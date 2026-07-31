# GISclaw product — cached descriptions of a project's data.
#
# The agent used to spend its first two or three rounds on list_files and
# inspect just to learn what it already had: field names, coordinate system,
# extent, how many features. That is the same answer every run, for data that
# rarely changes, paid for in rounds and tokens each time.
#
# So read it once, keep it beside the project, and hand it to the agent in the
# system prompt. The cache is keyed on each file's size and mtime, so editing
# or replacing a file re-reads only that file.
#
# What this is NOT: a substitute for the agent looking at the data. The profile
# carries schema and geometry facts, never values or statistics — the operating
# discipline still requires reading actual data before computing on it.

import json
import logging
import os
from typing import Optional

log = logging.getLogger("gisclaw.data_profile")

CACHE_NAME = ".data_profile.json"

VECTOR_EXT = {".shp", ".geojson", ".json", ".gpkg", ".gml", ".kml"}
RASTER_EXT = {".tif", ".tiff", ".img", ".vrt", ".nc"}
TABLE_EXT = {".csv", ".tsv", ".xlsx", ".xls"}

MAX_FIELDS = 40          # a very wide table shouldn't crowd out the prompt
MAX_FILES = 40


def _crs_label(crs) -> Optional[str]:
    """A short CRS name. The full WKT of a custom projection runs to hundreds
    of characters and would eat the prompt budget for no added meaning."""
    if crs is None:
        return None
    try:
        code = crs.to_epsg()
        if code:
            return f"EPSG:{code}"
    except Exception:
        pass
    try:
        return str(crs.name)[:80]
    except Exception:
        return str(crs)[:80]


def _key(path: str) -> str:
    st = os.stat(path)
    return f"{int(st.st_mtime)}:{st.st_size}"


def _profile_vector(path: str) -> dict:
    import geopandas as gpd
    gdf = gpd.read_file(path, rows=1)          # schema only — don't read the body
    try:
        import fiona
        with fiona.open(path) as src:
            count = len(src)
    except Exception:
        count = None
    fields = [c for c in gdf.columns if c != "geometry"][:MAX_FIELDS]
    out = {
        "kind": "vector",
        "features": count,
        "geometry": str(gdf.geom_type.iloc[0]) if len(gdf) else None,
        "crs": _crs_label(gdf.crs),
        "fields": fields,
        "dtypes": {c: str(gdf[c].dtype) for c in fields},
    }
    try:
        full = gpd.read_file(path)
        b = full.total_bounds
        out["bounds"] = [round(float(v), 4) for v in b]
        out["features"] = len(full)
    except Exception:
        pass
    return out


def _profile_raster(path: str) -> dict:
    import rasterio
    with rasterio.open(path) as s:
        b = s.bounds
        return {
            "kind": "raster",
            "bands": s.count,
            "size": [s.width, s.height],
            "dtype": str(s.dtypes[0]),
            "crs": _crs_label(s.crs),
            "nodata": None if s.nodata is None else float(s.nodata),
            "res": [abs(float(s.transform[0])), abs(float(s.transform[4]))],
            "bounds": [round(float(v), 4) for v in (b.left, b.bottom, b.right, b.top)],
        }


def _profile_table(path: str) -> dict:
    import pandas as pd
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, nrows=200)
    else:
        df = pd.read_csv(path, nrows=200, sep=None, engine="python")
    cols = list(df.columns)[:MAX_FIELDS]
    return {
        "kind": "table",
        "columns": cols,
        "dtypes": {c: str(df[c].dtype) for c in cols},
        "rows_sampled": len(df),
    }


def _profile_one(path: str) -> dict:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in VECTOR_EXT:
            return _profile_vector(path)
        if ext in RASTER_EXT:
            return _profile_raster(path)
        if ext in TABLE_EXT:
            return _profile_table(path)
    except Exception as e:
        # A file we can't read is worth saying so — the agent should know the
        # attempt failed rather than assume the file isn't there.
        return {"kind": "unreadable", "error": str(e)[:200]}
    return {"kind": "other"}


def profile_project(pdir: str) -> dict:
    """Profile everything in the project's data/, reusing unchanged entries."""
    data_dir = os.path.join(pdir, "data")
    cache_path = os.path.join(pdir, CACHE_NAME)
    cache = {}
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    out, changed = {}, False
    if os.path.isdir(data_dir):
        found = []
        for root, _dirs, files in os.walk(data_dir):
            for fn in sorted(files):
                if fn.startswith("."):
                    continue
                found.append(os.path.relpath(os.path.join(root, fn), data_dir))
        for rel in sorted(found)[:MAX_FILES]:
            full = os.path.join(data_dir, rel)
            try:
                k = _key(full)
            except OSError:
                continue
            hit = cache.get(rel)
            if hit and hit.get("_key") == k:
                out[rel] = hit
                continue
            prof = _profile_one(full)
            prof["_key"] = k
            out[rel] = prof
            changed = True
            log.info(f"profiled {rel}: {prof.get('kind')}")

    if changed or set(out) != set(cache):
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"could not write {CACHE_NAME}: {e}")
    return out


def build_block(profile: dict, char_cap: int = 3000) -> str:
    """Render the profile for the system prompt. Empty string if nothing to say."""
    if not profile:
        return ""
    lines = []
    for rel, p in profile.items():
        kind = p.get("kind")
        if kind == "vector":
            crs = p.get("crs") or "NONE DECLARED"
            head = (f"- {rel} — vector, {p.get('features')} features, "
                    f"{p.get('geometry')}, CRS {crs}")
            lines.append(head)
            if p.get("bounds"):
                lines.append(f"    bounds {p['bounds']}")
            if p.get("fields"):
                types = p.get("dtypes") or {}
                cols = ", ".join(f"{c} ({types.get(c, '?')})" for c in p["fields"])
                lines.append(f"    fields: {cols}")
        elif kind == "raster":
            crs = p.get("crs") or "NONE DECLARED"
            w, h = p.get("size", [None, None])
            lines.append(f"- {rel} — raster, {p.get('bands')} band(s), {w}x{h}, "
                         f"{p.get('dtype')}, CRS {crs}, nodata {p.get('nodata')}")
            if p.get("res"):
                lines.append(f"    pixel {p['res'][0]} x {p['res'][1]}, "
                             f"bounds {p.get('bounds')}")
        elif kind == "table":
            types = p.get("dtypes") or {}
            cols = ", ".join(f"{c} ({types.get(c, '?')})" for c in p.get("columns", []))
            lines.append(f"- {rel} — table, columns: {cols}")
        elif kind == "unreadable":
            lines.append(f"- {rel} — could not be opened: {p.get('error')}")
        else:
            lines.append(f"- {rel}")
    text = "\n".join(lines)
    if len(text) > char_cap:
        text = text[:char_cap].rsplit("\n", 1)[0] + "\n    … (truncated)"
    return text


def invalidate(pdir: str) -> None:
    """Drop the cache — call after data is added or removed."""
    try:
        os.remove(os.path.join(pdir, CACHE_NAME))
    except OSError:
        pass
