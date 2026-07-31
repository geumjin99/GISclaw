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

"""
geo_ops.py — deterministic GIS operation registry for the `geoprocess` tool.

Each operation is a fixed, tested code template (NOT LLM-generated) that runs in
the persistent PythonSandbox namespace. Operations read named variables the agent
already loaded (load_vector/load_raster) and bind their result back into the
namespace of tested, CRS-aware geoprocessing algorithms.

Design notes:
- Vector outputs auto-write `pred_results/<output>.geojson` in EPSG:4326 so the
  frontend renders the result on the map automatically (reuses the SSE result
  → map pipeline). The in-namespace variable keeps its own (working) CRS.
- Raster outputs write `pred_results/<output>.tif` and bind `<output>` (ndarray)
  + `<output>_meta` (dict), matching the load_raster convention.
- CRS footguns are handled inside the templates (e.g. buffer/distance ops
  reproject to an estimated UTM when the layer is geographic).

Registry entry: op_id -> {"category", "desc", "params", "build"}, where
build(inputs, params, output, save_as) -> python code string.
"""
from typing import Callable, Dict, Any


def _pv(x) -> str:
    """Render a JSON-ish value as a safe Python literal."""
    return repr(x)


# Boilerplate prepended to every op's code: helpers available to templates.
_PRELUDE = """
import os as _os, numpy as _np, geopandas as _gpd, pandas as _pd
_os.makedirs('pred_results', exist_ok=True)
def _disp_write(_g, _name):
    # Write a WGS84 GeoJSON snapshot for auto-render on the map.
    try:
        _d = _g
        if getattr(_d, 'crs', None) is not None:
            _d = _d.to_crs(4326)
        _d.to_file('pred_results/' + _name + '.geojson', driver='GeoJSON')
    except Exception:
        pass
def _summ(_g, _label):
    try:
        _crs = _g.crs.to_epsg() if getattr(_g, 'crs', None) is not None else None
        print(f"OK {_label}: {len(_g)} features, EPSG:{_crs}")
    except Exception:
        print(f"OK {_label}")
"""


def _vector_tail(output: str, save_as) -> str:
    """Common tail for vector ops: display snapshot + optional explicit save."""
    tail = f"_disp_write({output}, {_pv(output)})\n_summ({output}, {_pv(output)})\n"
    if save_as:
        tail += (f"try:\n    {output}.to_file({_pv(save_as)}, "
                 f"driver='GeoJSON' if {_pv(save_as)}.endswith('.geojson') else None)\n"
                 f"    print('saved -> ' + {_pv(save_as)})\nexcept Exception as _e:\n    print('save failed:', _e)\n")
    return tail


# ── CRS / projection ────────────────────────────────────────────────
def _b_reproject(inp, p, out, save_as):
    layer = inp["layer"]; crs = p["target_crs"]
    return _PRELUDE + f"{out} = {layer}.to_crs({_pv(crs)})\n" + _vector_tail(out, save_as)


def _b_set_crs(inp, p, out, save_as):
    layer = inp["layer"]; crs = p["crs"]
    return _PRELUDE + f"{out} = {layer}.set_crs({_pv(crs)}, allow_override=True)\n" + _vector_tail(out, save_as)


def _b_detect_crs(inp, p, out, save_as):
    layer = inp["layer"]
    return _PRELUDE + (
        f"_c = {layer}.crs\n"
        f"print('CRS:', _c)\n"
        f"print('EPSG:', _c.to_epsg() if _c is not None else None)\n"
        f"print('is_geographic:', _c.is_geographic if _c is not None else None)\n"
        f"print('bounds:', list({layer}.total_bounds))\n"
    )


# ── Vector geometry ─────────────────────────────────────────────────
def _b_buffer(inp, p, out, save_as):
    layer = inp["layer"]; dist = p["distance"]; dissolve = p.get("dissolve", False)
    code = _PRELUDE + (
        f"_src = {layer}\n"
        f"_geo = (_src.crs is None) or _src.crs.is_geographic\n"
        f"_work = _src.to_crs(_src.estimate_utm_crs()) if _geo else _src\n"   # buffer needs projected CRS
        f"{out} = _work.copy()\n"
        f"{out}['geometry'] = _work.geometry.buffer({_pv(dist)})\n"
    )
    if dissolve:
        code += f"{out} = {out}.dissolve().reset_index(drop=True)\n"
    code += f"{out} = {out}.to_crs(_src.crs) if _geo and _src.crs is not None else {out}\n"
    return code + _vector_tail(out, save_as)


def _b_centroid(inp, p, out, save_as):
    layer = inp["layer"]
    return _PRELUDE + (
        f"_src = {layer}\n"
        f"_geo = (_src.crs is None) or _src.crs.is_geographic\n"
        f"_work = _src.to_crs(_src.estimate_utm_crs()) if _geo else _src\n"
        f"{out} = _work.copy(); {out}['geometry'] = _work.geometry.centroid\n"
        f"{out} = {out}.to_crs(_src.crs) if _geo and _src.crs is not None else {out}\n"
    ) + _vector_tail(out, save_as)


def _b_convex_hull(inp, p, out, save_as):
    layer = inp["layer"]; by = p.get("by", "layer")
    if by == "feature":
        body = f"{out} = {layer}.copy(); {out}['geometry'] = {layer}.geometry.convex_hull\n"
    else:
        body = (f"from shapely.ops import unary_union as _uu\n"
                f"{out} = _gpd.GeoDataFrame(geometry=[_uu({layer}.geometry.values).convex_hull], crs={layer}.crs)\n")
    return _PRELUDE + body + _vector_tail(out, save_as)


def _b_dissolve(inp, p, out, save_as):
    layer = inp["layer"]; by = p.get("by")
    body = f"{out} = {layer}.dissolve(by={_pv(by)}).reset_index()\n" if by else f"{out} = {layer}.dissolve().reset_index(drop=True)\n"
    return _PRELUDE + body + _vector_tail(out, save_as)


def _b_simplify(inp, p, out, save_as):
    layer = inp["layer"]; tol = p["tolerance"]
    return _PRELUDE + (
        f"{out} = {layer}.copy(); {out}['geometry'] = {layer}.geometry.simplify({_pv(tol)}, preserve_topology=True)\n"
    ) + _vector_tail(out, save_as)


def _b_bounding_box(inp, p, out, save_as):
    layer = inp["layer"]
    return _PRELUDE + f"{out} = {layer}.copy(); {out}['geometry'] = {layer}.geometry.envelope\n" + _vector_tail(out, save_as)


def _b_explode(inp, p, out, save_as):
    layer = inp["layer"]
    return _PRELUDE + f"{out} = {layer}.explode(index_parts=False).reset_index(drop=True)\n" + _vector_tail(out, save_as)


# ── Vector overlay ──────────────────────────────────────────────────
def _overlay(how):
    def _b(inp, p, out, save_as):
        a = inp["layer"]; b = inp["overlay"]
        return _PRELUDE + (
            f"_b = {b}.to_crs({a}.crs) if ({a}.crs is not None and {b}.crs is not None and {a}.crs != {b}.crs) else {b}\n"
            f"{out} = _gpd.overlay({a}, _b, how={_pv(how)})\n"
        ) + _vector_tail(out, save_as)
    return _b


def _b_clip(inp, p, out, save_as):
    a = inp["layer"]; mask = inp["overlay"]
    return _PRELUDE + (
        f"_m = {mask}.to_crs({a}.crs) if ({a}.crs is not None and {mask}.crs is not None and {a}.crs != {mask}.crs) else {mask}\n"
        f"{out} = _gpd.clip({a}, _m)\n"
    ) + _vector_tail(out, save_as)


# ── Vector analysis / join ──────────────────────────────────────────
def _b_spatial_join(inp, p, out, save_as):
    a = inp["layer"]; b = inp["join_layer"]
    pred = p.get("predicate", "intersects"); how = p.get("how", "inner")
    return _PRELUDE + (
        f"_b = {b}.to_crs({a}.crs) if ({a}.crs is not None and {b}.crs is not None and {a}.crs != {b}.crs) else {b}\n"
        f"{out} = _gpd.sjoin({a}, _b, predicate={_pv(pred)}, how={_pv(how)})\n"
    ) + _vector_tail(out, save_as)


def _b_attribute_join(inp, p, out, save_as):
    a = inp["layer"]; b = inp["join_layer"]; on = p["on"]; how = p.get("how", "left")
    return _PRELUDE + (
        f"_bt = {b}.drop(columns='geometry') if 'geometry' in {b}.columns else {b}\n"
        f"{out} = {a}.merge(_bt, on={_pv(on)}, how={_pv(how)})\n"
    ) + _vector_tail(out, save_as)


def _b_count_in_polygon(inp, p, out, save_as):
    poly = inp["polygons"]; pts = inp["points"]; field = p.get("count_field", "NUMPOINTS")
    return _PRELUDE + (
        f"_pts = {pts}.to_crs({poly}.crs) if ({poly}.crs is not None and {pts}.crs is not None and {poly}.crs != {pts}.crs) else {pts}\n"
        f"_j = _gpd.sjoin({poly}, _pts, predicate='contains', how='left')\n"
        f"_cnt = _j.groupby(_j.index).size()\n"
        f"{out} = {poly}.copy(); {out}[{_pv(field)}] = {out}.index.map(_cnt).fillna(0).astype(int)\n"
    ) + _vector_tail(out, save_as)


def _b_select(inp, p, out, save_as):
    layer = inp["layer"]; expr = p["expression"]
    return _PRELUDE + f"{out} = {layer}.query({_pv(expr)}).reset_index(drop=True)\n" + _vector_tail(out, save_as)


def _b_add_field(inp, p, out, save_as):
    layer = inp["layer"]; name = p["name"]; expr = p["expression"]
    return _PRELUDE + (
        f"{out} = {layer}.copy()\n"
        f"{out}[{_pv(name)}] = {out}.eval({_pv(expr)})\n"
    ) + _vector_tail(out, save_as)


# ── Raster ──────────────────────────────────────────────────────────
def _b_zonal_statistics(inp, p, out, save_as):
    zones = inp["zones"]; raster = inp["raster"]
    stats = p.get("stats", ["mean", "min", "max", "count"])
    if isinstance(stats, str):
        stats = [s for s in stats.replace(",", " ").split() if s] or ["mean"]
    stats_str = " ".join(stats) if isinstance(stats, list) else str(stats)
    return _PRELUDE + (
        f"from rasterstats import zonal_stats as _zs\n"
        f"_arr = {raster}; _meta = {raster}_meta\n"
        f"_z = {zones}.to_crs(_meta['crs']) if ({zones}.crs is not None and _meta['crs'] is not None and {zones}.crs != _meta['crs']) else {zones}\n"
        f"_res = _zs(_z, _arr, affine=_meta['transform'], stats={_pv(stats_str)}, nodata=_meta.get('nodata'))\n"
        f"{out} = {zones}.copy()\n"
        f"for _k in {_pv(stats)}:\n"
        f"    {out}[_k] = [ _r.get(_k) for _r in _res ]\n"
    ) + _vector_tail(out, save_as)


def _raster_tail(out: str) -> str:
    """Write raster output to pred_results/<out>.tif and echo a summary."""
    return (
        f"import rasterio as _rio\n"
        f"_m = {out}_meta\n"
        f"with _rio.open('pred_results/{out}.tif', 'w', driver='GTiff', height={out}.shape[-2], "
        f"width={out}.shape[-1], count=1, dtype='float32', crs=_m['crs'], transform=_m['transform'], "
        f"nodata=_m.get('nodata')) as _dst:\n"
        f"    _dst.write({out}.astype('float32'), 1)\n"
        f"print(f'OK {out}: raster {{{out}.shape}}, EPSG:{{_m[\"crs\"].to_epsg() if _m[\"crs\"] else None}}')\n"
    )


def _b_clip_raster(inp, p, out, save_as):
    raster = inp["raster"]; mask = inp["mask"]
    return _PRELUDE + (
        f"import rasterio as _rio\n"
        f"from rasterio.features import geometry_mask as _gm\n"
        f"_arr = {raster}; _meta = dict({raster}_meta)\n"
        f"_mk = {mask}.to_crs(_meta['crs']) if ({mask}.crs is not None and _meta['crs'] is not None and {mask}.crs != _meta['crs']) else {mask}\n"
        f"_mask = _gm(_mk.geometry, out_shape=_arr.shape[-2:], transform=_meta['transform'], invert=True)\n"
        f"{out} = _np.where(_mask, _arr, _np.nan).astype('float32')\n"
        f"{out}_meta = _meta\n"
    ) + _raster_tail(out)


def _terrain(kind):
    def _b(inp, p, out, save_as):
        raster = inp["raster"]; zf = p.get("z_factor", 1.0)
        base = _PRELUDE + (
            f"_arr = {raster}.astype(float); _meta = dict({raster}_meta)\n"
            f"_px = abs(_meta['transform'].a); _py = abs(_meta['transform'].e)\n"
            f"_dy, _dx = _np.gradient(_arr * {_pv(zf)}, _py, _px)\n"
        )
        if kind == "slope":
            base += f"{out} = _np.degrees(_np.arctan(_np.sqrt(_dx**2 + _dy**2))).astype('float32')\n"
        elif kind == "aspect":
            base += f"{out} = (_np.degrees(_np.arctan2(-_dx, _dy)) % 360).astype('float32')\n"
        else:  # hillshade
            base += (
                f"_az, _alt = _np.radians(315.0), _np.radians(45.0)\n"
                f"_slope = _np.arctan(_np.sqrt(_dx**2 + _dy**2))\n"
                f"_aspect = _np.arctan2(-_dx, _dy)\n"
                f"{out} = (255*((_np.cos(_alt)*_np.cos(_slope)) + (_np.sin(_alt)*_np.sin(_slope)*_np.cos(_az-_aspect)))).clip(0,255).astype('float32')\n"
            )
        base += f"{out}_meta = _meta\n"
        return base + _raster_tail(out)
    return _b


def _b_rasterize(inp, p, out, save_as):
    layer = inp["layer"]; res = p["resolution"]; field = p.get("field")
    return _PRELUDE + (
        f"import rasterio as _rio\n"
        f"from rasterio import features as _feat\n"
        f"from rasterio.transform import from_origin as _fo\n"
        f"_g = {layer}\n"
        f"_minx,_miny,_maxx,_maxy = _g.total_bounds\n"
        f"_w = max(1,int((_maxx-_minx)/{_pv(res)})); _h = max(1,int((_maxy-_miny)/{_pv(res)}))\n"
        f"_tr = _fo(_minx,_maxy,{_pv(res)},{_pv(res)})\n"
        f"_shapes = ((_geom, (_val if {_pv(field)} else 1)) for _geom,_val in zip(_g.geometry, (_g[{_pv(field)}] if {_pv(field)} else [1]*len(_g))))\n"
        f"{out} = _feat.rasterize(_shapes, out_shape=(_h,_w), transform=_tr, fill=0, dtype='float32')\n"
        f"{out}_meta = dict(crs=_g.crs, transform=_tr, width=_w, height=_h, nodata=0, dtype='float32', count=1)\n"
    ) + _raster_tail(out)


def _b_polygonize(inp, p, out, save_as):
    raster = inp["raster"]
    return _PRELUDE + (
        f"from rasterio import features as _feat\n"
        f"from shapely.geometry import shape as _shape\n"
        f"_arr = {raster}; _meta = {raster}_meta\n"
        f"_a = _arr if _arr.ndim==2 else _arr[0]\n"
        f"_valid = ~_np.isnan(_a)\n"
        f"_recs = [(_shape(_gj), _v) for _gj,_v in _feat.shapes(_np.nan_to_num(_a).astype('float32'), mask=_valid, transform=_meta['transform'])]\n"
        f"{out} = _gpd.GeoDataFrame({{'value':[_v for _,_v in _recs]}}, geometry=[_g for _g,_ in _recs], crs=_meta['crs'])\n"
    ) + _vector_tail(out, save_as)


def _b_sample_raster(inp, p, out, save_as):
    raster = inp["raster"]; pts = inp["points"]; field = p.get("field", "value")
    return _PRELUDE + (
        f"_arr = {raster}; _meta = {raster}_meta\n"
        f"_p = {pts}.to_crs(_meta['crs']) if ({pts}.crs is not None and _meta['crs'] is not None and {pts}.crs != _meta['crs']) else {pts}\n"
        f"_a = _arr if _arr.ndim==2 else _arr[0]\n"
        f"_inv = ~_meta['transform']\n"
        f"_vals = []\n"
        f"for _geom in _p.geometry:\n"
        f"    _col,_row = _inv * (_geom.x, _geom.y)\n"
        f"    _r,_c = int(_row), int(_col)\n"
        f"    _vals.append(float(_a[_r,_c]) if (0<=_r<_a.shape[0] and 0<=_c<_a.shape[1]) else _np.nan)\n"
        f"{out} = {pts}.copy(); {out}[{_pv(field)}] = _vals\n"
    ) + _vector_tail(out, save_as)


def _b_idw(inp, p, out, save_as):
    pts = inp["points"]; field = p["field"]; res = p["resolution"]; power = p.get("power", 2)
    return _PRELUDE + (
        f"import rasterio as _rio\n"
        f"from rasterio.transform import from_origin as _fo\n"
        f"_p = {pts}\n"
        f"_minx,_miny,_maxx,_maxy = _p.total_bounds\n"
        f"_w = max(1,int((_maxx-_minx)/{_pv(res)})); _h = max(1,int((_maxy-_miny)/{_pv(res)}))\n"
        f"_gx = _np.linspace(_minx+{_pv(res)}/2, _maxx-{_pv(res)}/2, _w)\n"
        f"_gy = _np.linspace(_maxy-{_pv(res)}/2, _miny+{_pv(res)}/2, _h)\n"
        f"_GX,_GY = _np.meshgrid(_gx,_gy)\n"
        f"_px = _np.array([g.x for g in _p.geometry]); _py = _np.array([g.y for g in _p.geometry])\n"
        f"_pv_ = _p[{_pv(field)}].astype(float).values\n"
        f"_num = _np.zeros((_h,_w)); _den = _np.zeros((_h,_w))\n"
        f"for _xi,_yi,_vi in zip(_px,_py,_pv_):\n"
        f"    _d = _np.sqrt((_GX-_xi)**2 + (_GY-_yi)**2); _d[_d==0]=1e-9\n"
        f"    _wgt = 1.0/_d**{_pv(power)}; _num += _wgt*_vi; _den += _wgt\n"
        f"{out} = (_num/_den).astype('float32')\n"
        f"_tr = _fo(_minx,_maxy,{_pv(res)},{_pv(res)})\n"
        f"{out}_meta = dict(crs=_p.crs, transform=_tr, width=_w, height=_h, nodata=None, dtype='float32', count=1)\n"
    ) + _raster_tail(out)


# ── Registry ────────────────────────────────────────────────────────
# category, one-line desc, params help, build fn
REGISTRY: Dict[str, Dict[str, Any]] = {
    # CRS
    "reproject":       {"cat": "crs", "desc": "Reproject a vector layer to a target CRS", "params": "target_crs (EPSG int or 'EPSG:xxxx')", "build": _b_reproject},
    "set_crs":         {"cat": "crs", "desc": "Assign a CRS without transforming coordinates", "params": "crs", "build": _b_set_crs},
    "detect_crs":      {"cat": "crs", "desc": "Report a layer's CRS/EPSG/bounds", "params": "-", "build": _b_detect_crs},
    # geometry
    "buffer":          {"cat": "geometry", "desc": "Buffer geometries by a distance (auto-uses a projected CRS)", "params": "distance, dissolve(bool)", "build": _b_buffer},
    "centroid":        {"cat": "geometry", "desc": "Geometric centroids", "params": "-", "build": _b_centroid},
    "convex_hull":     {"cat": "geometry", "desc": "Convex hull (by feature or whole layer)", "params": "by: feature|layer", "build": _b_convex_hull},
    "dissolve":        {"cat": "geometry", "desc": "Dissolve/merge features, optionally by a field", "params": "by(field, optional)", "build": _b_dissolve},
    "simplify":        {"cat": "geometry", "desc": "Simplify geometries (Douglas-Peucker)", "params": "tolerance", "build": _b_simplify},
    "bounding_box":    {"cat": "geometry", "desc": "Per-feature bounding boxes (envelopes)", "params": "-", "build": _b_bounding_box},
    "explode":         {"cat": "geometry", "desc": "Multipart to singleparts", "params": "-", "build": _b_explode},
    # overlay
    "clip":            {"cat": "overlay", "desc": "Clip layer by a mask layer", "params": "inputs.overlay = mask layer", "build": _b_clip},
    "intersection":    {"cat": "overlay", "desc": "Intersection of two layers", "params": "inputs.overlay", "build": _overlay("intersection")},
    "difference":      {"cat": "overlay", "desc": "Difference (layer minus overlay)", "params": "inputs.overlay", "build": _overlay("difference")},
    "union":           {"cat": "overlay", "desc": "Union of two layers", "params": "inputs.overlay", "build": _overlay("union")},
    # analysis / join
    "spatial_join":    {"cat": "analysis", "desc": "Join attributes by spatial relationship", "params": "inputs.join_layer; predicate, how", "build": _b_spatial_join},
    "attribute_join":  {"cat": "analysis", "desc": "Join attributes by a common field", "params": "inputs.join_layer; on, how", "build": _b_attribute_join},
    "count_in_polygon":{"cat": "analysis", "desc": "Count points within each polygon", "params": "inputs.polygons, inputs.points; count_field", "build": _b_count_in_polygon},
    "select":          {"cat": "analysis", "desc": "Select/filter features by an expression (pandas query)", "params": "expression", "build": _b_select},
    "add_field":       {"cat": "analysis", "desc": "Add a computed field (pandas eval)", "params": "name, expression", "build": _b_add_field},
    # raster
    "zonal_statistics":{"cat": "raster", "desc": "Summarize raster values within vector zones", "params": "inputs.zones, inputs.raster; stats(list)", "build": _b_zonal_statistics},
    "clip_raster":     {"cat": "raster", "desc": "Clip a raster by a vector mask", "params": "inputs.raster, inputs.mask", "build": _b_clip_raster},
    "slope":           {"cat": "raster", "desc": "Slope (degrees) from a DEM", "params": "inputs.raster; z_factor", "build": _terrain("slope")},
    "aspect":          {"cat": "raster", "desc": "Aspect (degrees) from a DEM", "params": "inputs.raster; z_factor", "build": _terrain("aspect")},
    "hillshade":       {"cat": "raster", "desc": "Hillshade from a DEM", "params": "inputs.raster; z_factor", "build": _terrain("hillshade")},
    "rasterize":       {"cat": "raster", "desc": "Rasterize a vector layer", "params": "resolution; field(optional)", "build": _b_rasterize},
    "polygonize":      {"cat": "raster", "desc": "Convert raster pixels/regions to polygons", "params": "inputs.raster", "build": _b_polygonize},
    "sample_raster":   {"cat": "raster", "desc": "Sample raster values at point locations", "params": "inputs.raster, inputs.points; field", "build": _b_sample_raster},
    "idw":             {"cat": "raster", "desc": "IDW interpolation from points to a raster", "params": "inputs.points; field, resolution, power", "build": _b_idw},
}


def build_code(op: str, inputs: dict, params: dict, output: str, save_as=None) -> str:
    """Return the deterministic code string for an operation."""
    if op not in REGISTRY:
        raise KeyError(op)
    return REGISTRY[op]["build"](inputs or {}, params or {}, output, save_as)


def describe_ops() -> str:
    """Compact, grouped op catalog for the tool description / system prompt."""
    cats = {}
    for op, meta in REGISTRY.items():
        cats.setdefault(meta["cat"], []).append((op, meta["desc"], meta["params"]))
    order = ["crs", "geometry", "overlay", "analysis", "raster"]
    lines = []
    for c in order:
        lines.append(f"  [{c}]")
        for op, desc, params in cats.get(c, []):
            lines.append(f"    - {op}: {desc}. params: {params}")
    return "\n".join(lines)


# ── Structured specs (for the UI Toolbox: input roles + parameter forms) ──
# in:   list of (role, kind)  kind = "vector" | "raster"
# args: list of (name, type, default, required)  type = number|text|bool|select|crs
#       select args carry choices via the CHOICES map below.
def _I(*pairs):
    return [{"role": r, "kind": k} for r, k in pairs]


def _A(*specs):
    out = []
    for s in specs:
        name, typ, default, required = s
        out.append({"name": name, "type": typ, "default": default, "required": required})
    return out


CHOICES = {
    "by": ["layer", "feature"],
    "predicate": ["intersects", "contains", "within", "crosses", "overlaps"],
    "how": ["inner", "left"],
}

SPECS: Dict[str, Dict[str, Any]] = {
    "reproject":        {"in": _I(("layer", "vector")), "args": _A(("target_crs", "crs", 4326, True))},
    "set_crs":          {"in": _I(("layer", "vector")), "args": _A(("crs", "crs", 4326, True))},
    "detect_crs":       {"in": _I(("layer", "vector")), "args": _A()},
    "buffer":           {"in": _I(("layer", "vector")), "args": _A(("distance", "number", 100, True), ("dissolve", "bool", False, False))},
    "centroid":         {"in": _I(("layer", "vector")), "args": _A()},
    "convex_hull":      {"in": _I(("layer", "vector")), "args": _A(("by", "select", "layer", False))},
    "dissolve":         {"in": _I(("layer", "vector")), "args": _A(("by", "text", "", False))},
    "simplify":         {"in": _I(("layer", "vector")), "args": _A(("tolerance", "number", 0.001, True))},
    "bounding_box":     {"in": _I(("layer", "vector")), "args": _A()},
    "explode":          {"in": _I(("layer", "vector")), "args": _A()},
    "clip":             {"in": _I(("layer", "vector"), ("overlay", "vector")), "args": _A()},
    "intersection":     {"in": _I(("layer", "vector"), ("overlay", "vector")), "args": _A()},
    "difference":       {"in": _I(("layer", "vector"), ("overlay", "vector")), "args": _A()},
    "union":            {"in": _I(("layer", "vector"), ("overlay", "vector")), "args": _A()},
    "spatial_join":     {"in": _I(("layer", "vector"), ("join_layer", "vector")), "args": _A(("predicate", "select", "intersects", False), ("how", "select", "inner", False))},
    "attribute_join":   {"in": _I(("layer", "vector"), ("join_layer", "vector")), "args": _A(("on", "text", "", True), ("how", "select", "left", False))},
    "count_in_polygon": {"in": _I(("polygons", "vector"), ("points", "vector")), "args": _A(("count_field", "text", "NUMPOINTS", False))},
    "select":           {"in": _I(("layer", "vector")), "args": _A(("expression", "text", "", True))},
    "add_field":        {"in": _I(("layer", "vector")), "args": _A(("name", "text", "", True), ("expression", "text", "", True))},
    "zonal_statistics": {"in": _I(("zones", "vector"), ("raster", "raster")), "args": _A(("stats", "text", "mean min max count", False))},
    "clip_raster":      {"in": _I(("raster", "raster"), ("mask", "vector")), "args": _A()},
    "slope":            {"in": _I(("raster", "raster")), "args": _A(("z_factor", "number", 1.0, False))},
    "aspect":           {"in": _I(("raster", "raster")), "args": _A(("z_factor", "number", 1.0, False))},
    "hillshade":        {"in": _I(("raster", "raster")), "args": _A(("z_factor", "number", 1.0, False))},
    "rasterize":        {"in": _I(("layer", "vector")), "args": _A(("resolution", "number", 100, True), ("field", "text", "", False))},
    "polygonize":       {"in": _I(("raster", "raster")), "args": _A()},
    "sample_raster":    {"in": _I(("raster", "raster"), ("points", "vector")), "args": _A(("field", "text", "value", False))},
    "idw":              {"in": _I(("points", "vector")), "args": _A(("field", "text", "", True), ("resolution", "number", 100, True), ("power", "number", 2, False))},
}


def catalog() -> list:
    """Grouped op catalog with structured specs, for the UI Toolbox."""
    order = ["crs", "geometry", "overlay", "analysis", "raster"]
    groups = {c: [] for c in order}
    for op, meta in REGISTRY.items():
        spec = SPECS.get(op, {"in": [], "args": []})
        entry = {
            "op": op, "desc": meta["desc"],
            "inputs": spec["in"],
            "args": [dict(a, choices=CHOICES.get(a["name"])) if a["type"] == "select" else a
                     for a in spec["args"]],
        }
        groups.setdefault(meta["cat"], []).append(entry)
    return [{"category": c, "ops": groups[c]} for c in order if groups.get(c)]
