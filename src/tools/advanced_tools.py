"""
(Advanced analysis tools)

GIS 

"""
import os
from typing import Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from .registry import tool_registry
from .vector_tools import get_loaded_datasets, _loaded_datasets

def _get_utm_epsg(gdf):
    """ UTM EPSG """
    # UTM 
    if gdf.crs and gdf.crs.is_projected:
        return gdf.crs.to_epsg() or 3857
    centroid = gdf.geometry.unary_union.centroid
    utm_zone = int((centroid.x + 180) / 6) + 1
    hemisphere = "north" if centroid.y >= 0 else "south"
    return 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone

# -- 1: --------------------------
@tool_registry.register(
    name="group_by_count",
    description=(
        "Group features by a column and count the number of features in each group. "
        "Returns a summary table with group names and counts, sorted by count descending."
    ),
    parameters={
        "dataset_name": {
            "type": "string",
            "description": "Name of the dataset"
        },
        "column": {
            "type": "string",
            "description": "Column name to group by"
        },
    },
    returns="Table of groups with counts",
)
def group_by_count(dataset_name: str, column: str) -> str:
    """"""
    try:
        datasets = get_loaded_datasets()
        gdf = datasets.get(dataset_name)
        if gdf is None:
            return f"Error: Dataset '{dataset_name}' not found."
        if column not in gdf.columns:
            return f"Error: Column '{column}' not found. Available: {list(gdf.columns)}"

        counts = gdf[column].value_counts().reset_index()
        counts.columns = [column, "count"]

        lines = [f"Group-by '{column}' on '{dataset_name}' ({len(gdf)} features):"]
        lines.append(f"  Unique groups: {len(counts)}")
        lines.append(f"  {'Group':<30s} {'Count':>8s} {'%':>7s}")
        lines.append(f"  {'-'*47}")
        for _, row in counts.head(20).iterrows():
            pct = row["count"] / len(gdf) * 100
            lines.append(f"  {str(row[column]):<30s} {row['count']:>8d} {pct:>6.1f}%")
        if len(counts) > 20:
            lines.append(f"  ... and {len(counts) - 20} more groups")

        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"

# -- 2: ----------------------
@tool_registry.register(
    name="calculate_length",
    description=(
        "Calculate the length of line features in a dataset. "
        "Adds 'length_m' (meters) and 'length_km' (kilometers) columns. "
        "Also reports total length and average length."
    ),
    parameters={
        "dataset_name": {
            "type": "string",
            "description": "Name of the line dataset"
        },
        "output_name": {
            "type": "string",
            "description": "Name for the output dataset with length columns"
        },
    },
    returns="Summary of length calculations",
)
def calculate_length(dataset_name: str, output_name: str) -> str:
    """"""
    try:
        datasets = get_loaded_datasets()
        gdf = datasets.get(dataset_name)
        if gdf is None:
            return f"Error: Dataset '{dataset_name}' not found."

        epsg = _get_utm_epsg(gdf)
        gdf_proj = gdf.to_crs(epsg=epsg)

        result = gdf.copy()
        result["length_m"] = gdf_proj.geometry.length.values
        result["length_km"] = result["length_m"] / 1000
        _loaded_datasets[output_name] = result

        total_km = result["length_km"].sum()
        return (
            f"Length calculation complete.\n"
            f"  Features: {len(result)}\n"
            f"  Total length: {total_km:.2f} km\n"
            f"  Range: {result['length_km'].min():.4f} - {result['length_km'].max():.4f} km\n"
            f"  Mean: {result['length_km'].mean():.4f} km\n"
            f"  Saved as '{output_name}' with columns 'length_m' and 'length_km'."
        )
    except Exception as e:
        return f"Error: {str(e)}"

# -- 3: ----------------------
@tool_registry.register(
    name="create_point",
    description=(
        "Create a point dataset from longitude and latitude coordinates. "
        "Useful for defining specific locations like earthquake epicenters, "
        "construction sites, or reference points for analysis."
    ),
    parameters={
        "longitude": {
            "type": "number",
            "description": "Longitude (X) coordinate in decimal degrees"
        },
        "latitude": {
            "type": "number",
            "description": "Latitude (Y) coordinate in decimal degrees"
        },
        "output_name": {
            "type": "string",
            "description": "Name for the output point dataset"
        },
        "label": {
            "type": "string",
            "description": "Optional label/name for the point. Use 'none' to skip.",
            "optional": True,
        },
    },
    returns="Confirmation of point creation",
)
def create_point(
    longitude: float, latitude: float, output_name: str, label: str = "none"
) -> str:
    """"""
    try:
        lon = float(longitude)
        lat = float(latitude)
        point = Point(lon, lat)
        data = {"name": [label if label != "none" else output_name], "geometry": [point]}
        gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        _loaded_datasets[output_name] = gdf

        return (
            f"Point created at ({lon:.6f}, {lat:.6f}).\n"
            f"  Label: '{label if label != 'none' else output_name}'\n"
            f"  CRS: EPSG:4326\n"
            f"  Saved as '{output_name}'."
        )
    except Exception as e:
        return f"Error: {str(e)}"

# -- 4: --------------------------
@tool_registry.register(
    name="dissolve",
    description=(
        "Dissolve (merge) geometries by a shared attribute column. "
        "Features with the same value in the specified column will be merged into one geometry. "
        "Useful for aggregating individual features into zones or regions."
    ),
    parameters={
        "dataset_name": {
            "type": "string",
            "description": "Name of the input dataset"
        },
        "by_column": {
            "type": "string",
            "description": "Column to dissolve by. Use 'all' to merge all features into one."
        },
        "output_name": {
            "type": "string",
            "description": "Name for the output dissolved dataset"
        },
    },
    returns="Summary of dissolve result",
)
def dissolve(dataset_name: str, by_column: str, output_name: str) -> str:
    """"""
    try:
        datasets = get_loaded_datasets()
        gdf = datasets.get(dataset_name)
        if gdf is None:
            return f"Error: Dataset '{dataset_name}' not found."

        if by_column == "all":
            dissolved = gdf.dissolve()
        else:
            if by_column not in gdf.columns:
                return f"Error: Column '{by_column}' not found. Available: {list(gdf.columns)}"
            dissolved = gdf.dissolve(by=by_column)

        dissolved = dissolved.reset_index()
        _loaded_datasets[output_name] = dissolved

        return (
            f"Dissolve complete.\n"
            f"  Input: {len(gdf)} features from '{dataset_name}'\n"
            f"  Dissolved by: '{by_column}'\n"
            f"  Result: {len(dissolved)} merged features\n"
            f"  Saved as '{output_name}'."
        )
    except Exception as e:
        return f"Error: {str(e)}"

# -- 5: --------------------------
@tool_registry.register(
    name="get_statistics",
    description=(
        "Calculate descriptive statistics for a numeric column in a dataset. "
        "Returns count, min, max, mean, median, std, sum, and percentiles."
    ),
    parameters={
        "dataset_name": {
            "type": "string",
            "description": "Name of the dataset"
        },
        "column": {
            "type": "string",
            "description": "Numeric column to calculate statistics for"
        },
    },
    returns="Statistical summary of the column",
)
def get_statistics(dataset_name: str, column: str) -> str:
    """"""
    try:
        datasets = get_loaded_datasets()
        gdf = datasets.get(dataset_name)
        if gdf is None:
            return f"Error: Dataset '{dataset_name}' not found."
        if column not in gdf.columns:
            return f"Error: Column '{column}' not found. Available: {list(gdf.columns)}"

        values = pd.to_numeric(gdf[column], errors="coerce")
        valid = values.dropna()

        if len(valid) == 0:
            return f"Error: Column '{column}' has no numeric values."

        return (
            f"Statistics for '{column}' in '{dataset_name}':\n"
            f"  Count: {len(valid)} (of {len(gdf)} total, {len(gdf)-len(valid)} non-numeric)\n"
            f"  Min: {valid.min():.4f}\n"
            f"  Max: {valid.max():.4f}\n"
            f"  Mean: {valid.mean():.4f}\n"
            f"  Median: {valid.median():.4f}\n"
            f"  Std: {valid.std():.4f}\n"
            f"  Sum: {valid.sum():.4f}\n"
            f"  25th percentile: {valid.quantile(0.25):.4f}\n"
            f"  75th percentile: {valid.quantile(0.75):.4f}"
        )
    except Exception as e:
        return f"Error: {str(e)}"

# -- 6: --------------------------
@tool_registry.register(
    name="merge_datasets",
    description=(
        "Merge (concatenate) two datasets into one. "
        "Both datasets must have the same CRS. "
        "Useful for combining data from different sources or cities."
    ),
    parameters={
        "dataset_a": {
            "type": "string",
            "description": "Name of the first dataset"
        },
        "dataset_b": {
            "type": "string",
            "description": "Name of the second dataset"
        },
        "output_name": {
            "type": "string",
            "description": "Name for the merged output dataset"
        },
    },
    returns="Summary of merged dataset",
)
def merge_datasets(dataset_a: str, dataset_b: str, output_name: str) -> str:
    """"""
    try:
        datasets = get_loaded_datasets()
        a = datasets.get(dataset_a)
        b = datasets.get(dataset_b)
        if a is None:
            return f"Error: Dataset '{dataset_a}' not found."
        if b is None:
            return f"Error: Dataset '{dataset_b}' not found."

        # CRS 
        if a.crs != b.crs:
            b = b.to_crs(a.crs)

        merged = pd.concat([a, b], ignore_index=True)
        merged = gpd.GeoDataFrame(merged, crs=a.crs)
        _loaded_datasets[output_name] = merged

        return (
            f"Merge complete.\n"
            f"  '{dataset_a}': {len(a)} features\n"
            f"  '{dataset_b}': {len(b)} features\n"
            f"  Result: {len(merged)} features\n"
            f"  Saved as '{output_name}'."
        )
    except Exception as e:
        return f"Error: {str(e)}"

# -- 7: ------------------------------
@tool_registry.register(
    name="clip",
    description=(
        "Clip a dataset to the boundary of another dataset. "
        "Only features (or parts of features) that fall within the "
        "clip boundary are retained."
    ),
    parameters={
        "input_dataset": {
            "type": "string",
            "description": "Name of the dataset to clip"
        },
        "clip_dataset": {
            "type": "string",
            "description": "Name of the polygon dataset to use as clip boundary"
        },
        "output_name": {
            "type": "string",
            "description": "Name for the clipped output dataset"
        },
    },
    returns="Summary of clip result",
)
def clip(input_dataset: str, clip_dataset: str, output_name: str) -> str:
    """"""
    try:
        datasets = get_loaded_datasets()
        inp = datasets.get(input_dataset)
        clp = datasets.get(clip_dataset)
        if inp is None:
            return f"Error: Dataset '{input_dataset}' not found."
        if clp is None:
            return f"Error: Dataset '{clip_dataset}' not found."

        # CRS 
        if inp.crs != clp.crs:
            clp = clp.to_crs(inp.crs)

        result = gpd.clip(inp, clp)
        _loaded_datasets[output_name] = result

        return (
            f"Clip complete.\n"
            f"  Input: '{input_dataset}' ({len(inp)} features)\n"
            f"  Clip boundary: '{clip_dataset}' ({len(clp)} features)\n"
            f"  Result: {len(result)} features retained\n"
            f"  Saved as '{output_name}'."
        )
    except Exception as e:
        return f"Error: {str(e)}"

# -- 8: --------------------------
@tool_registry.register(
    name="add_column",
    description=(
        "Add a new computed column to a dataset. "
        "Supported computations: 'area_km2' (polygon area), 'length_km' (line length), "
        "'centroid_x' (X coordinate), 'centroid_y' (Y coordinate), "
        "'density' (feature count per km² based on area column). "
        "The dataset is modified in place."
    ),
    parameters={
        "dataset_name": {
            "type": "string",
            "description": "Name of the dataset"
        },
        "computation": {
            "type": "string",
            "description": "Type of computation: 'area_km2', 'length_km', 'centroid_x', 'centroid_y'"
        },
        "output_name": {
            "type": "string",
            "description": "Name for the output dataset"
        },
    },
    returns="Confirmation with summary of the new column",
)
def add_column(dataset_name: str, computation: str, output_name: str) -> str:
    """"""
    try:
        datasets = get_loaded_datasets()
        gdf = datasets.get(dataset_name)
        if gdf is None:
            return f"Error: Dataset '{dataset_name}' not found."

        result = gdf.copy()
        valid_computations = ["area_km2", "length_km", "centroid_x", "centroid_y"]

        if computation == "area_km2":
            epsg = _get_utm_epsg(gdf)
            proj = gdf.to_crs(epsg=epsg)
            result["area_km2"] = proj.geometry.area.values / 1e6
            col_name = "area_km2"
        elif computation == "length_km":
            epsg = _get_utm_epsg(gdf)
            proj = gdf.to_crs(epsg=epsg)
            result["length_km"] = proj.geometry.length.values / 1000
            col_name = "length_km"
        elif computation == "centroid_x":
            result["centroid_x"] = gdf.geometry.centroid.x
            col_name = "centroid_x"
        elif computation == "centroid_y":
            result["centroid_y"] = gdf.geometry.centroid.y
            col_name = "centroid_y"
        else:
            return f"Error: Unknown computation '{computation}'. Use: {valid_computations}"

        _loaded_datasets[output_name] = result
        values = result[col_name]

        return (
            f"Column '{col_name}' added to '{dataset_name}'.\n"
            f"  Range: {values.min():.6f} - {values.max():.6f}\n"
            f"  Mean: {values.mean():.6f}\n"
            f"  Saved as '{output_name}'."
        )
    except Exception as e:
        return f"Error: {str(e)}"
