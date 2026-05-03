"""
(Spatial Analysis Tools)

"""
import os
from typing import Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from .registry import tool_registry
from .vector_tools import get_loaded_datasets, _loaded_datasets

@tool_registry.register(
    name="buffer_analysis",
    description=(
        "Create buffer zones around features in a dataset. "
        "The buffer distance is specified in meters. "
        "Input data is automatically reprojected to a metric CRS for accurate distance calculation."
    ),
    parameters={
        "dataset_name": {
            "type": "string",
            "description": "Name of the input dataset"
        },
        "distance_meters": {
            "type": "number",
            "description": "Buffer distance in meters"
        },
        "output_name": {
            "type": "string",
            "description": "Name for the output buffered dataset"
        },
    },
    returns="Summary of the buffer result",
)
def buffer_analysis(dataset_name: str, distance_meters: float, output_name: str) -> str:
    """"""
    try:
        datasets = get_loaded_datasets()
        gdf = datasets.get(dataset_name)
        if gdf is None:
            return f"Error: Dataset '{dataset_name}' not found."

        # CRS (UTM) 
        original_crs = gdf.crs
        if gdf.crs and gdf.crs.is_projected:
            epsg = gdf.crs.to_epsg() or 3857
        else:
            centroid = gdf.geometry.unary_union.centroid
            utm_zone = int((centroid.x + 180) / 6) + 1
            hemisphere = "north" if centroid.y >= 0 else "south"
            epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone

        gdf_proj = gdf.to_crs(epsg=epsg)
        buffered = gdf_proj.copy()
        buffered["geometry"] = gdf_proj.geometry.buffer(distance_meters)

        # CRS
        buffered = buffered.to_crs(original_crs)
        _loaded_datasets[output_name] = buffered

        total_area_km2 = gdf_proj.geometry.buffer(distance_meters).area.sum() / 1e6
        return (
            f"Buffer analysis complete.\n"
            f"  Input: {len(gdf)} features from '{dataset_name}'\n"
            f"  Buffer distance: {distance_meters}m\n"
            f"  Total buffered area: {total_area_km2:.4f} km²\n"
            f"  Saved as '{output_name}'."
        )
    except Exception as e:
        return f"Error: {str(e)}"

@tool_registry.register(
    name="spatial_join",
    description=(
        "Perform a spatial join between two datasets. "
        "Finds features from the right dataset that intersect with features in the left dataset. "
        "Supports 'intersects', 'contains', and 'within' predicates."
    ),
    parameters={
        "left_dataset": {
            "type": "string",
            "description": "Name of the left (base) dataset"
        },
        "right_dataset": {
            "type": "string",
            "description": "Name of the right (join) dataset"
        },
        "predicate": {
            "type": "string",
            "description": "Spatial predicate: 'intersects', 'contains', or 'within'"
        },
        "output_name": {
            "type": "string",
            "description": "Name for the output joined dataset"
        },
    },
    returns="Summary of the spatial join result",
)
def spatial_join(
    left_dataset: str, right_dataset: str, predicate: str, output_name: str
) -> str:
    """"""
    try:
        datasets = get_loaded_datasets()
        left = datasets.get(left_dataset)
        right = datasets.get(right_dataset)

        if left is None:
            return f"Error: Dataset '{left_dataset}' not found."
        if right is None:
            return f"Error: Dataset '{right_dataset}' not found."

        valid_predicates = ["intersects", "contains", "within"]
        if predicate not in valid_predicates:
            return f"Error: Invalid predicate '{predicate}'. Use: {valid_predicates}"

        # CRS 
        if left.crs != right.crs:
            right = right.to_crs(left.crs)

        # index_right / index_left join 
        for col in ['index_right', 'index_left']:
            if col in left.columns:
                left = left.drop(columns=[col])
            if col in right.columns:
                right = right.drop(columns=[col])

        result = gpd.sjoin(left, right, how="inner", predicate=predicate)
        _loaded_datasets[output_name] = result

        unique_left = result.index.nunique()
        unique_right = result["index_right"].nunique() if "index_right" in result.columns else "?"

        return (
            f"Spatial join complete.\n"
            f"  Left: '{left_dataset}' ({len(left)} features)\n"
            f"  Right: '{right_dataset}' ({len(right)} features)\n"
            f"  Predicate: {predicate}\n"
            f"  Result: {len(result)} matched rows\n"
            f"  Note: {unique_left} unique features from '{left_dataset}' matched "
            f"(one left feature may match multiple right features, causing duplicate rows).\n"
            f"  Saved as '{output_name}'."
        )

    except Exception as e:
        return f"Error: {str(e)}"

@tool_registry.register(
    name="overlay_analysis",
    description=(
        "Perform overlay analysis (intersection, union, difference) between two polygon datasets. "
        "Useful for finding areas that overlap or differ between two sets of polygons."
    ),
    parameters={
        "left_dataset": {
            "type": "string",
            "description": "Name of the first polygon dataset"
        },
        "right_dataset": {
            "type": "string",
            "description": "Name of the second polygon dataset"
        },
        "how": {
            "type": "string",
            "description": "Overlay method: 'intersection', 'union', 'difference', 'symmetric_difference'"
        },
        "output_name": {
            "type": "string",
            "description": "Name for the output dataset"
        },
    },
    returns="Summary of the overlay result",
)
def overlay_analysis(
    left_dataset: str, right_dataset: str, how: str, output_name: str
) -> str:
    """"""
    try:
        datasets = get_loaded_datasets()
        left = datasets.get(left_dataset)
        right = datasets.get(right_dataset)

        if left is None:
            return f"Error: Dataset '{left_dataset}' not found."
        if right is None:
            return f"Error: Dataset '{right_dataset}' not found."

        valid_methods = ["intersection", "union", "difference", "symmetric_difference"]
        if how not in valid_methods:
            return f"Error: Invalid method '{how}'. Use: {valid_methods}"

        # CRS 
        if left.crs != right.crs:
            right = right.to_crs(left.crs)

        result = gpd.overlay(left, right, how=how)
        _loaded_datasets[output_name] = result

        return (
            f"Overlay analysis complete.\n"
            f"  Left: '{left_dataset}' ({len(left)} features)\n"
            f"  Right: '{right_dataset}' ({len(right)} features)\n"
            f"  Method: {how}\n"
            f"  Result: {len(result)} features\n"
            f"  Saved as '{output_name}'."
        )
    except Exception as e:
        return f"Error: {str(e)}"

@tool_registry.register(
    name="calculate_distance",
    description=(
        "Calculate the distance (in meters) from each feature in the source dataset "
        "to the nearest feature in the target dataset. "
        "Adds a 'distance_m' column to the source dataset."
    ),
    parameters={
        "source_dataset": {
            "type": "string",
            "description": "Name of the source dataset (distance FROM these features)"
        },
        "target_dataset": {
            "type": "string",
            "description": "Name of the target dataset (distance TO nearest of these features)"
        },
        "output_name": {
            "type": "string",
            "description": "Name for the output dataset with distance column"
        },
    },
    returns="Summary of distance calculations",
)
def calculate_distance(source_dataset: str, target_dataset: str, output_name: str) -> str:
    """"""
    try:
        datasets = get_loaded_datasets()
        source = datasets.get(source_dataset)
        target = datasets.get(target_dataset)

        if source is None:
            return f"Error: Dataset '{source_dataset}' not found."
        if target is None:
            return f"Error: Dataset '{target_dataset}' not found."

        MAX_ROWS = 10000
        sampled = False
        if len(source) > MAX_ROWS:
            source = source.sample(n=MAX_ROWS, random_state=42)
            sampled = True

        # UTM unary_union
        c = source.geometry.iloc[0].centroid
        utm_zone = int((c.x + 180) / 6) + 1
        hemisphere = "north" if c.y >= 0 else "south"
        epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone

        source_proj = source.to_crs(epsg=epsg)
        target_proj = target.to_crs(epsg=epsg)

        target_union = target_proj.geometry.unary_union

        distances = source_proj.geometry.distance(target_union)

        result = source.copy()
        result["distance_m"] = distances.values
        _loaded_datasets[output_name] = result

        note = f" (sampled from {source_dataset})" if sampled else ""
        return (
            f"Distance calculation complete.{note}\n"
            f"  Source: '{source_dataset}' ({len(source)} features)\n"
            f"  Target: '{target_dataset}' ({len(target)} features)\n"
            f"  Distance range: {distances.min():.1f}m - {distances.max():.1f}m\n"
            f"  Mean distance: {distances.mean():.1f}m\n"
            f"  Saved as '{output_name}' with column 'distance_m'."
        )
    except Exception as e:
        return f"Error: {str(e)}"

@tool_registry.register(
    name="calculate_area",
    description=(
        "Calculate the area (in square meters and square kilometers) of polygon features. "
        "Adds 'area_m2' and 'area_km2' columns."
    ),
    parameters={
        "dataset_name": {
            "type": "string",
            "description": "Name of the polygon dataset"
        },
        "output_name": {
            "type": "string",
            "description": "Name for the output dataset with area columns"
        },
    },
    returns="Summary of area calculations",
)
def calculate_area(dataset_name: str, output_name: str) -> str:
    """"""
    try:
        datasets = get_loaded_datasets()
        gdf = datasets.get(dataset_name)
        if gdf is None:
            return f"Error: Dataset '{dataset_name}' not found."

        # CRS
        if gdf.crs and gdf.crs.is_projected:
            epsg = gdf.crs.to_epsg() or 3857
        else:
            centroid = gdf.geometry.unary_union.centroid
            utm_zone = int((centroid.x + 180) / 6) + 1
            hemisphere = "north" if centroid.y >= 0 else "south"
            epsg = 32600 + utm_zone if hemisphere == "north" else 32700 + utm_zone

        gdf_proj = gdf.to_crs(epsg=epsg)

        result = gdf.copy()
        result["area_m2"] = gdf_proj.geometry.area.values
        result["area_km2"] = result["area_m2"] / 1e6
        _loaded_datasets[output_name] = result

        total_km2 = result["area_km2"].sum()
        return (
            f"Area calculation complete.\n"
            f"  Features: {len(result)}\n"
            f"  Total area: {total_km2:.4f} km²\n"
            f"  Range: {result['area_km2'].min():.6f} - {result['area_km2'].max():.4f} km²\n"
            f"  Mean: {result['area_km2'].mean():.6f} km²\n"
            f"  Saved as '{output_name}' with columns 'area_m2' and 'area_km2'."
        )
    except Exception as e:
        return f"Error: {str(e)}"
