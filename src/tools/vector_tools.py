"""
(Vector Tools)

Shapefile, GeoJSON
"""
import os
import json
from typing import Optional, Dict, Any, List

import geopandas as gpd
import pandas as pd

from .registry import tool_registry

# Agent 
_loaded_datasets: Dict[str, gpd.GeoDataFrame] = {}

def _get_dataset(dataset_name: str) -> gpd.GeoDataFrame:
    """"""
    if dataset_name not in _loaded_datasets:
        available = list(_loaded_datasets.keys()) if _loaded_datasets else ["(none)"]
        raise ValueError(
            f"Dataset '{dataset_name}' not found. "
            f"Available datasets: {available}. "
            f"Please use load_vector first."
        )
    return _loaded_datasets[dataset_name]

@tool_registry.register(
    name="load_vector",
    description=(
        "Load a vector geospatial dataset from a file path. "
        "Supports Shapefile, GeoJSON, GPKG, CSV/TSV (with coordinate columns). "
        "For GPKG with multiple layers, use the 'layer' parameter. "
        "For CSV/TSV, coordinate columns (Latitude/Longitude, x/y, etc.) are auto-detected."
    ),
    parameters={
        "file_path": {
            "type": "string",
            "description": "Path to the vector data file (e.g., 'data/seoul/buildings.geojson')"
        },
        "dataset_name": {
            "type": "string",
            "description": "A short name to reference this dataset later (e.g., 'buildings')"
        },
        "layer": {
            "type": "string",
            "description": "(Optional) Layer name for multi-layer files like GPKG. If not specified, the first layer is loaded.",
            "optional": True
        },
    },
    returns="Summary string with row count, columns, CRS, and geometry type",
)
def load_vector(file_path: str, dataset_name: str, layer: str = "") -> str:
    """ SHP/GeoJSON/GPKG/CSV/TSV"""
    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"

    try:
        ext = os.path.splitext(file_path)[1].lower()

        # CSV/TSV GeoDataFrame
        if ext in (".csv", ".tsv", ".txt"):
            sep = "\t" if ext in (".tsv", ".txt") else ","
            df = pd.read_csv(file_path, sep=sep)

            lon_candidates = ["longitude", "lon", "long", "x", "x coordinate", "x_coordinate", "lng", "intptlong"]
            lat_candidates = ["latitude", "lat", "y", "y coordinate", "y_coordinate", "intptlat"]

            col_lower_map = {c.lower().strip(): c for c in df.columns}
            lon_col = next((col_lower_map[k] for k in lon_candidates if k in col_lower_map), None)
            lat_col = next((col_lower_map[k] for k in lat_candidates if k in col_lower_map), None)

            if lon_col and lat_col:
                df = df.dropna(subset=[lon_col, lat_col])
                df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
                df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
                df = df.dropna(subset=[lon_col, lat_col])

                from shapely.geometry import Point
                geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
                gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            else:
                return (
                    f"Error: Could not find coordinate columns in CSV/TSV.\n"
                    f"  Available columns: {list(df.columns)}\n"
                    f"  Expected longitude columns: {lon_candidates}\n"
                    f"  Expected latitude columns: {lat_candidates}"
                )

        # GPKG 
        elif ext == ".gpkg":
            import fiona
            available_layers = fiona.listlayers(file_path)
            if layer:
                if layer not in available_layers:
                    return (
                        f"Error: Layer '{layer}' not found in GPKG.\n"
                        f"  Available layers: {available_layers}"
                    )
                gdf = gpd.read_file(file_path, layer=layer)
            else:
                gdf = gpd.read_file(file_path, layer=available_layers[0])
                if len(available_layers) > 1:
                    layer_info = f"\n  Note: GPKG has {len(available_layers)} layers: {available_layers}. Loaded first layer '{available_layers[0]}'."
                else:
                    layer_info = ""

        # SHP, GeoJSON 
        else:
            gdf = gpd.read_file(file_path)

        _loaded_datasets[dataset_name] = gdf

        summary = (
            f"Dataset '{dataset_name}' loaded successfully.\n"
            f"  Rows: {len(gdf)}\n"
            f"  Columns: {list(gdf.columns)}\n"
            f"  CRS: {gdf.crs}\n"
            f"  Geometry type: {gdf.geometry.geom_type.unique().tolist()}\n"
            f"  Bounds: {gdf.total_bounds.tolist()}"
        )

        # GPKG 
        if ext == ".gpkg" and not layer and len(available_layers) > 1:
            summary += layer_info

        return summary
    except Exception as e:
        return f"Error loading file: {str(e)}"

@tool_registry.register(
    name="get_dataset_info",
    description=(
        "Get detailed information about a loaded dataset, including column names, "
        "data types, sample values, and basic statistics for numeric columns."
    ),
    parameters={
        "dataset_name": {
            "type": "string",
            "description": "Name of the previously loaded dataset"
        },
    },
    returns="Detailed dataset information string",
)
def get_dataset_info(dataset_name: str) -> str:
    """"""
    try:
        gdf = _get_dataset(dataset_name)

        col_info = []
        for col in gdf.columns:
            if col == "geometry":
                col_info.append(f"  - geometry ({gdf.geometry.geom_type.unique().tolist()})")
            else:
                dtype = str(gdf[col].dtype)
                sample = str(gdf[col].iloc[0]) if len(gdf) > 0 else "N/A"
                if len(sample) > 50:
                    sample = sample[:50] + "..."
                col_info.append(f"  - {col} ({dtype}): sample='{sample}'")

        info = (
            f"Dataset '{dataset_name}':\n"
            f"  Rows: {len(gdf)}\n"
            f"  CRS: {gdf.crs}\n"
            f"  Columns:\n" + "\n".join(col_info)
        )

        numeric_cols = gdf.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            stats = gdf[numeric_cols].describe().to_string()
            info += f"\n\n  Numeric statistics:\n{stats}"

        return info
    except Exception as e:
        return f"Error: {str(e)}"

@tool_registry.register(
    name="filter_by_attribute",
    description=(
        "Filter a loaded dataset by an attribute condition and save as a new dataset. "
        "Supports comparison operators: ==, !=, >, <, >=, <=, contains."
    ),
    parameters={
        "dataset_name": {
            "type": "string",
            "description": "Name of the source dataset"
        },
        "column": {
            "type": "string",
            "description": "Column name to filter on"
        },
        "operator": {
            "type": "string",
            "description": "Comparison operator: '==', '!=', '>', '<', '>=', '<=', 'contains'"
        },
        "value": {
            "type": "string",
            "description": "Value to compare against (will be auto-converted to appropriate type)"
        },
        "output_name": {
            "type": "string",
            "description": "Name for the filtered result dataset"
        },
    },
    returns="Summary of the filtered dataset",
)
def filter_by_attribute(
    dataset_name: str, column: str, operator: str, value: str, output_name: str
) -> str:
    """"""
    try:
        gdf = _get_dataset(dataset_name)

        if column not in gdf.columns:
            return f"Error: Column '{column}' not found. Available: {list(gdf.columns)}"

        col_dtype = gdf[column].dtype
        try:
            if pd.api.types.is_numeric_dtype(col_dtype):
                typed_value = float(value)
            else:
                typed_value = value
        except ValueError:
            typed_value = value

        ops = {
            "==": lambda s, v: s == v,
            "!=": lambda s, v: s != v,
            ">": lambda s, v: s > v,
            "<": lambda s, v: s < v,
            ">=": lambda s, v: s >= v,
            "<=": lambda s, v: s <= v,
            "contains": lambda s, v: s.astype(str).str.contains(str(v), case=False, na=False),
        }

        if operator not in ops:
            return f"Error: Unknown operator '{operator}'. Supported: {list(ops.keys())}"

        mask = ops[operator](gdf[column], typed_value)
        filtered = gdf[mask].copy()
        _loaded_datasets[output_name] = filtered

        return (
            f"Filtered '{dataset_name}' where {column} {operator} '{value}': "
            f"{len(filtered)} rows (from {len(gdf)} original). "
            f"Saved as '{output_name}'."
        )
    except Exception as e:
        return f"Error: {str(e)}"

@tool_registry.register(
    name="count_features",
    description="Count the number of features (rows) in a loaded dataset.",
    parameters={
        "dataset_name": {
            "type": "string",
            "description": "Name of the dataset to count"
        },
    },
    returns="Number of features as a string",
)
def count_features(dataset_name: str) -> str:
    """"""
    try:
        gdf = _get_dataset(dataset_name)
        return f"Dataset '{dataset_name}' has {len(gdf)} features."
    except Exception as e:
        return f"Error: {str(e)}"

@tool_registry.register(
    name="save_vector",
    description="Save a loaded dataset to a file (GeoJSON format).",
    parameters={
        "dataset_name": {
            "type": "string",
            "description": "Name of the dataset to save"
        },
        "output_path": {
            "type": "string",
            "description": "Output file path (e.g., 'output/result.geojson')"
        },
    },
    returns="Confirmation message with output path",
)
def save_vector(dataset_name: str, output_path: str) -> str:
    """"""
    try:
        gdf = _get_dataset(dataset_name)
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        gdf.to_file(output_path, driver="GeoJSON")
        return f"Dataset '{dataset_name}' saved to '{output_path}' ({len(gdf)} features)."
    except Exception as e:
        return f"Error: {str(e)}"

def get_loaded_datasets() -> Dict[str, gpd.GeoDataFrame]:
    """"""
    return _loaded_datasets

def clear_datasets():
    """"""
    _loaded_datasets.clear()
