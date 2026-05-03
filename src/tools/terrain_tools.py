"""
(Terrain Analysis Tools)

rasterio + numpy QGIS/GDAL CLI
"""
import os
import json
import numpy as np
from typing import Optional, Dict, Any

import rasterio
from rasterio.transform import from_bounds
from rasterio.features import shapes as rio_shapes

from .registry import tool_registry
from .vector_tools import _loaded_datasets

_loaded_rasters: Dict[str, Dict[str, Any]] = {}

def _get_raster(name: str):
    """"""
    # raster_tools 
    try:
        from .raster_tools import _raster_store
        if name in _raster_store:
            return _raster_store[name]
    except ImportError:
        pass
    if name in _loaded_rasters:
        return _loaded_rasters[name]
    raise ValueError(f"Raster '{name}' not found. Load it first with load_raster.")

def _ensure_output_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

# -- 1: --
@tool_registry.register(
    name="calculate_slope",
    description="Calculate slope (in degrees) from a DEM raster. Returns statistics and saves to file.",
    parameters={
        "raster_name": {"type": "string", "description": "Name of loaded DEM raster"},
        "output_file": {"type": "string", "description": "Output file path for slope raster"},
    },
    returns="Slope statistics (min, max, mean) and output file path",
)
def calculate_slope(raster_name: str, output_file: str = "output/slope.tif"):
    """"""
    rinfo = _get_raster(raster_name)
    data = rinfo["data"]
    transform = rinfo["transform"]
    crs = rinfo["crs"]
    nodata = rinfo.get("nodata", -9999)

    # numpy gradient 
    if data.ndim == 3:
        dem = data[0].astype(float)
    else:
        dem = data.astype(float)

    res_x = abs(transform[0])
    res_y = abs(transform[4])

    dy, dx = np.gradient(dem, res_y, res_x)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    slope_deg = np.degrees(slope_rad)

    # nodata
    if nodata is not None:
        mask = dem == nodata
        slope_deg[mask] = nodata

    _ensure_output_dir(output_file)
    valid = slope_deg[slope_deg != nodata] if nodata else slope_deg
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": dem.shape[1], "height": dem.shape[0],
        "count": 1, "crs": crs, "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(slope_deg.astype(np.float32), 1)

    stats = {
        "min_degrees": round(float(np.nanmin(valid)), 2),
        "max_degrees": round(float(np.nanmax(valid)), 2),
        "mean_degrees": round(float(np.nanmean(valid)), 2),
        "output_file": output_file,
    }
    return json.dumps(stats)

# -- 2: --
@tool_registry.register(
    name="calculate_aspect",
    description="Calculate aspect (direction of slope, in degrees 0-360) from a DEM raster.",
    parameters={
        "raster_name": {"type": "string", "description": "Name of loaded DEM raster"},
        "output_file": {"type": "string", "description": "Output file path"},
    },
    returns="Aspect statistics and output file path",
)
def calculate_aspect(raster_name: str, output_file: str = "output/aspect.tif"):
    """0-360"""
    rinfo = _get_raster(raster_name)
    data = rinfo["data"]
    transform = rinfo["transform"]
    crs = rinfo["crs"]
    nodata = rinfo.get("nodata", -9999)

    dem = data[0].astype(float) if data.ndim == 3 else data.astype(float)
    res_x, res_y = abs(transform[0]), abs(transform[4])

    dy, dx = np.gradient(dem, res_y, res_x)
    aspect = np.degrees(np.arctan2(-dy, dx))
    # (0=North, 90=East)
    aspect = (90 - aspect) % 360

    if nodata is not None:
        mask = dem == nodata
        aspect[mask] = nodata

    _ensure_output_dir(output_file)
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": dem.shape[1], "height": dem.shape[0],
        "count": 1, "crs": crs, "transform": transform, "nodata": nodata,
    }
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(aspect.astype(np.float32), 1)

    valid = aspect[aspect != nodata] if nodata else aspect
    return json.dumps({
        "min": round(float(np.nanmin(valid)), 1),
        "max": round(float(np.nanmax(valid)), 1),
        "mean": round(float(np.nanmean(valid)), 1),
        "output_file": output_file,
    })

# -- 3: --
@tool_registry.register(
    name="generate_hillshade",
    description="Generate hillshade visualization from a DEM raster.",
    parameters={
        "raster_name": {"type": "string", "description": "Name of loaded DEM raster"},
        "azimuth": {"type": "number", "description": "Sun azimuth angle (default 315)", "optional": True},
        "altitude": {"type": "number", "description": "Sun altitude angle (default 45)", "optional": True},
        "output_file": {"type": "string", "description": "Output file path"},
    },
    returns="Hillshade output file path",
)
def generate_hillshade(raster_name: str, output_file: str = "output/hillshade.tif",
                       azimuth: float = 315.0, altitude: float = 45.0):
    """"""
    rinfo = _get_raster(raster_name)
    data = rinfo["data"]
    transform = rinfo["transform"]

    dem = data[0].astype(float) if data.ndim == 3 else data.astype(float)
    res_x, res_y = abs(transform[0]), abs(transform[4])

    dy, dx = np.gradient(dem, res_y, res_x)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)

    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)

    hillshade = (np.sin(alt_rad) * np.cos(slope) +
                 np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    hillshade = np.clip(hillshade * 255, 0, 255).astype(np.uint8)

    _ensure_output_dir(output_file)
    profile = {
        "driver": "GTiff", "dtype": "uint8",
        "width": dem.shape[1], "height": dem.shape[0],
        "count": 1, "crs": rinfo["crs"], "transform": transform,
    }
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(hillshade, 1)

    return json.dumps({"output_file": output_file, "shape": list(hillshade.shape)})

# -- 4: --
@tool_registry.register(
    name="generate_contour",
    description="Generate contour lines from a DEM raster at specified interval.",
    parameters={
        "raster_name": {"type": "string", "description": "Name of loaded DEM raster"},
        "interval": {"type": "number", "description": "Contour interval (meters)"},
        "output_file": {"type": "string", "description": "Output GeoJSON file path"},
    },
    returns="Number of contour lines and output file path",
)
def generate_contour(raster_name: str, interval: float, output_file: str = "output/contours.geojson"):
    """"""
    import geopandas as gpd
    from shapely.geometry import LineString, mapping
    from matplotlib import pyplot as plt

    rinfo = _get_raster(raster_name)
    data = rinfo["data"]
    transform = rinfo["transform"]

    dem = data[0].astype(float) if data.ndim == 3 else data.astype(float)
    nodata = rinfo.get("nodata")
    if nodata is not None:
        dem[dem == nodata] = np.nan

    vmin, vmax = np.nanmin(dem), np.nanmax(dem)
    levels = np.arange(np.floor(vmin / interval) * interval, vmax + interval, interval)

    # matplotlib contour 
    fig, ax = plt.subplots()
    cs = ax.contour(dem, levels=levels)
    plt.close(fig)

    features = []
    for i, level in enumerate(cs.levels):
        for path in cs.collections[i].get_paths():
            coords = path.vertices
            xs = transform[2] + coords[:, 0] * transform[0]
            ys = transform[5] + coords[:, 1] * transform[4]
            if len(xs) >= 2:
                features.append({"geometry": LineString(zip(xs, ys)), "elevation": float(level)})

    gdf = gpd.GeoDataFrame(features, crs=rinfo.get("crs", "EPSG:4326"))
    _ensure_output_dir(output_file)
    gdf.to_file(output_file, driver="GeoJSON")

    return json.dumps({
        "contour_lines": len(features),
        "interval": interval,
        "elevation_range": [round(float(vmin), 1), round(float(vmax), 1)],
        "output_file": output_file,
    })

# -- 5: --
@tool_registry.register(
    name="raster_calculator",
    description="Perform mathematical operations on raster bands. Supports expressions like 'A-B', '(A-B)/(A+B)' for NDVI, etc. Use 'A' for first raster, 'B' for second.",
    parameters={
        "raster_a": {"type": "string", "description": "Name of first raster (referenced as 'A' in expression)"},
        "band_a": {"type": "integer", "description": "Band index for A (1-based)", "optional": True},
        "raster_b": {"type": "string", "description": "Name of second raster (referenced as 'B')", "optional": True},
        "band_b": {"type": "integer", "description": "Band index for B (1-based)", "optional": True},
        "expression": {"type": "string", "description": "Math expression using A and B, e.g. '(A-B)/(A+B)'"},
        "output_file": {"type": "string", "description": "Output file path"},
    },
    returns="Statistics of result (min, max, mean) and output file path",
)
def raster_calculator(raster_a: str, expression: str,
                      band_a: int = 1, raster_b: str = "", band_b: int = 1,
                      output_file: str = "output/raster_calc.tif"):
    """"""
    ra = _get_raster(raster_a)
    A = ra["data"][band_a - 1].astype(float) if ra["data"].ndim == 3 else ra["data"].astype(float)

    B = None
    if raster_b:
        rb = _get_raster(raster_b)
        B = rb["data"][band_b - 1].astype(float) if rb["data"].ndim == 3 else rb["data"].astype(float)

    with np.errstate(divide='ignore', invalid='ignore'):
        result = eval(expression, {"__builtins__": {}, "np": np, "A": A, "B": B})

    result = np.where(np.isfinite(result), result, np.nan)

    _ensure_output_dir(output_file)
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": A.shape[1], "height": A.shape[0],
        "count": 1, "crs": ra["crs"], "transform": ra["transform"],
    }
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(result.astype(np.float32), 1)

    valid = result[np.isfinite(result)]
    return json.dumps({
        "expression": expression,
        "min": round(float(np.nanmin(valid)), 4) if len(valid) > 0 else None,
        "max": round(float(np.nanmax(valid)), 4) if len(valid) > 0 else None,
        "mean": round(float(np.nanmean(valid)), 4) if len(valid) > 0 else None,
        "output_file": output_file,
    })

# -- 6: --
@tool_registry.register(
    name="calculate_roughness",
    description="Calculate terrain roughness index (TRI) from a DEM raster.",
    parameters={
        "raster_name": {"type": "string", "description": "Name of loaded DEM raster"},
        "output_file": {"type": "string", "description": "Output file path"},
    },
    returns="Roughness statistics and output file path",
)
def calculate_roughness(raster_name: str, output_file: str = "output/roughness.tif"):
    """ (TRI)"""
    rinfo = _get_raster(raster_name)
    data = rinfo["data"]
    dem = data[0].astype(float) if data.ndim == 3 else data.astype(float)

    # TRI: 8
    from scipy.ndimage import generic_filter

    def tri_func(window):
        center = window[4]
        return np.sqrt(np.sum((window - center)**2) / 8)

    tri = generic_filter(dem, tri_func, size=3, mode='nearest')

    _ensure_output_dir(output_file)
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": dem.shape[1], "height": dem.shape[0],
        "count": 1, "crs": rinfo["crs"], "transform": rinfo["transform"],
    }
    with rasterio.open(output_file, "w", **profile) as dst:
        dst.write(tri.astype(np.float32), 1)

    return json.dumps({
        "min": round(float(np.nanmin(tri)), 2),
        "max": round(float(np.nanmax(tri)), 2),
        "mean": round(float(np.nanmean(tri)), 2),
        "output_file": output_file,
    })

# -- 7: --
@tool_registry.register(
    name="reproject_raster",
    description="Reproject a raster to a different coordinate reference system.",
    parameters={
        "raster_name": {"type": "string", "description": "Name of loaded raster"},
        "target_crs": {"type": "string", "description": "Target CRS, e.g. 'EPSG:32617'"},
        "output_file": {"type": "string", "description": "Output file path"},
    },
    returns="Reprojected raster info",
)
def reproject_raster(raster_name: str, target_crs: str, output_file: str = "output/reprojected.tif"):
    """"""
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.crs import CRS

    rinfo = _get_raster(raster_name)
    src_crs = CRS.from_user_input(rinfo["crs"]) if rinfo["crs"] else CRS.from_epsg(4326)
    dst_crs = CRS.from_user_input(target_crs)
    data = rinfo["data"]

    if data.ndim == 2:
        data = data[np.newaxis, :, :]

    transform, width, height = calculate_default_transform(
        src_crs, dst_crs, data.shape[2], data.shape[1],
        *rasterio.transform.array_bounds(data.shape[1], data.shape[2], rinfo["transform"])
    )

    _ensure_output_dir(output_file)
    profile = {
        "driver": "GTiff", "dtype": str(data.dtype),
        "width": width, "height": height,
        "count": data.shape[0], "crs": dst_crs, "transform": transform,
    }
    with rasterio.open(output_file, "w", **profile) as dst:
        for i in range(data.shape[0]):
            reproject(
                source=data[i], destination=rasterio.band(dst, i + 1),
                src_transform=rinfo["transform"], src_crs=src_crs,
                dst_transform=transform, dst_crs=dst_crs,
                resampling=Resampling.bilinear,
            )

    return json.dumps({
        "source_crs": str(src_crs), "target_crs": str(dst_crs),
        "width": width, "height": height,
        "output_file": output_file,
    })
