"""
(Raster Tools)

GeoTIFF
"""
import os
from typing import Dict, Any, Optional

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
import geopandas as gpd
from rasterstats import zonal_stats as _zonal_stats

from .registry import tool_registry
from .vector_tools import get_loaded_datasets, _loaded_datasets

_loaded_rasters: Dict[str, str] = {}  # name -> file_path
# terrain_tools 
_raster_store: Dict[str, Dict[str, Any]] = {}  # name -> {data, transform, crs, nodata}

@tool_registry.register(
    name="load_raster",
    description=(
        "Load a raster dataset (GeoTIFF) and get its metadata. "
        "Returns information about dimensions, CRS, resolution, and value statistics."
    ),
    parameters={
        "file_path": {
            "type": "string",
            "description": "Path to the raster file (e.g., 'data/seoul/dem.tif')"
        },
        "raster_name": {
            "type": "string",
            "description": "A short name to reference this raster later (e.g., 'dem')"
        },
    },
    returns="Raster metadata summary string",
)
def load_raster(file_path: str, raster_name: str) -> str:
    """"""
    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"

    try:
        with rasterio.open(file_path) as src:
            data = src.read()
            single = src.read(1)
            nodata = src.nodata
            if nodata is not None:
                valid_data = single[single != nodata]
            else:
                valid_data = single[~np.isnan(single)] if np.issubdtype(single.dtype, np.floating) else single.flatten()

            _loaded_rasters[raster_name] = file_path
            # terrain_tools 
            _raster_store[raster_name] = {
                "data": data,
                "transform": src.transform,
                "crs": str(src.crs) if src.crs else None,
                "nodata": nodata,
                "width": src.width,
                "height": src.height,
                "bands": src.count,
            }

            summary = (
                f"Raster '{raster_name}' loaded successfully.\n"
                f"  Shape: {src.width} x {src.height} pixels\n"
                f"  Bands: {src.count}\n"
                f"  CRS: {src.crs}\n"
                f"  Resolution: {src.res[0]:.6f} x {src.res[1]:.6f}\n"
                f"  Bounds: {list(src.bounds)}\n"
                f"  Value range: {valid_data.min():.2f} - {valid_data.max():.2f}\n"
                f"  Mean: {valid_data.mean():.2f}, Std: {valid_data.std():.2f}"
            )
            return summary
    except Exception as e:
        return f"Error loading raster: {str(e)}"

@tool_registry.register(
    name="zonal_statistics",
    description=(
        "Calculate zonal statistics of a raster within zones defined by a vector dataset. "
        "Computes min, max, mean, median, sum, count, and std for each zone. "
        "Results are added as new columns to the vector dataset."
    ),
    parameters={
        "vector_dataset": {
            "type": "string",
            "description": "Name of the vector dataset defining zones"
        },
        "raster_name": {
            "type": "string",
            "description": "Name of the raster dataset to analyze"
        },
        "stats_prefix": {
            "type": "string",
            "description": "Prefix for new column names (e.g., 'elevation')"
        },
        "output_name": {
            "type": "string",
            "description": "Name for the output dataset with statistics"
        },
    },
    returns="Summary of computed statistics",
)
def zonal_statistics(
    vector_dataset: str, raster_name: str, stats_prefix: str, output_name: str
) -> str:
    """"""
    try:
        datasets = get_loaded_datasets()
        gdf = datasets.get(vector_dataset)
        if gdf is None:
            return f"Error: Vector dataset '{vector_dataset}' not found."

        raster_path = _loaded_rasters.get(raster_name)
        if raster_path is None:
            return f"Error: Raster '{raster_name}' not found. Load it first with load_raster."

        # CRS 
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs

        if gdf.crs != raster_crs:
            gdf_proj = gdf.to_crs(raster_crs)
        else:
            gdf_proj = gdf

        stats = _zonal_stats(
            gdf_proj, raster_path,
            stats=["min", "max", "mean", "median", "sum", "count", "std"],
        )

        result = gdf.copy()
        for stat_name in ["min", "max", "mean", "median", "sum", "count", "std"]:
            col_name = f"{stats_prefix}_{stat_name}"
            result[col_name] = [s[stat_name] if s[stat_name] is not None else np.nan for s in stats]

        _loaded_datasets[output_name] = result

        mean_col = f"{stats_prefix}_mean"
        if mean_col in result.columns:
            valid = result[mean_col].dropna()
            summary = (
                f"Zonal statistics computed for {len(result)} zones.\n"
                f"  New columns: {[f'{stats_prefix}_{s}' for s in ['min','max','mean','median','sum','count','std']]}\n"
                f"  {mean_col} range: {valid.min():.2f} - {valid.max():.2f}\n"
                f"  {mean_col} overall mean: {valid.mean():.2f}\n"
                f"  Saved as '{output_name}'."
            )
        else:
            summary = f"Zonal statistics computed and saved as '{output_name}'."

        return summary
    except Exception as e:
        return f"Error: {str(e)}"

@tool_registry.register(
    name="get_raster_value_at_points",
    description=(
        "Extract raster values at point locations from a vector dataset. "
        "Adds a new column with the raster value at each point."
    ),
    parameters={
        "point_dataset": {
            "type": "string",
            "description": "Name of the point vector dataset"
        },
        "raster_name": {
            "type": "string",
            "description": "Name of the raster to sample from"
        },
        "value_column": {
            "type": "string",
            "description": "Name for the new column containing raster values"
        },
        "output_name": {
            "type": "string",
            "description": "Name for the output dataset"
        },
    },
    returns="Summary of extracted values",
)
def get_raster_value_at_points(
    point_dataset: str, raster_name: str, value_column: str, output_name: str
) -> str:
    """"""
    try:
        datasets = get_loaded_datasets()
        gdf = datasets.get(point_dataset)
        if gdf is None:
            return f"Error: Dataset '{point_dataset}' not found."

        raster_path = _loaded_rasters.get(raster_name)
        if raster_path is None:
            return f"Error: Raster '{raster_name}' not found."

        with rasterio.open(raster_path) as src:
            # CRS 
            if gdf.crs != src.crs:
                gdf_proj = gdf.to_crs(src.crs)
            else:
                gdf_proj = gdf

            coords = [(geom.x, geom.y) for geom in gdf_proj.geometry]
            values = list(src.sample(coords))
            values = [v[0] for v in values]

        result = gdf.copy()
        result[value_column] = values
        _loaded_datasets[output_name] = result

        valid_values = [v for v in values if v is not None and not np.isnan(v)]
        return (
            f"Extracted raster values for {len(result)} points.\n"
            f"  New column: '{value_column}'\n"
            f"  Valid values: {len(valid_values)}/{len(values)}\n"
            f"  Range: {min(valid_values):.2f} - {max(valid_values):.2f}\n"
            f"  Mean: {np.mean(valid_values):.2f}\n"
            f"  Saved as '{output_name}'."
        )
    except Exception as e:
        return f"Error: {str(e)}"

def get_loaded_rasters() -> Dict[str, str]:
    """"""
    return _loaded_rasters

def clear_rasters():
    """"""
    _loaded_rasters.clear()
    _raster_store.clear()
