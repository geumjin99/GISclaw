"""
(Conversion Tools)

↔
"""
import os
import json
import numpy as np
from typing import Optional

import geopandas as gpd
import rasterio
from rasterio.crs import CRS

from .registry import tool_registry
from .vector_tools import _loaded_datasets

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

# -- --
@tool_registry.register(
    name="reproject_vector",
    description="Reproject a vector dataset to a different coordinate reference system.",
    parameters={
        "dataset_name": {"type": "string", "description": "Dataset to reproject"},
        "target_crs": {"type": "string", "description": "Target CRS (e.g. 'EPSG:32617')"},
        "output_name": {"type": "string", "description": "Name for reprojected dataset"},
    },
    returns="Reprojected dataset info",
)
def reproject_vector(dataset_name: str, target_crs: str, output_name: str):
    """"""
    gdf = _loaded_datasets.get(dataset_name)
    if gdf is None:
        return json.dumps({"error": f"Dataset '{dataset_name}' not found"})

    reprojected = gdf.to_crs(target_crs)
    _loaded_datasets[output_name] = reprojected

    return json.dumps({
        "dataset_name": output_name,
        "source_crs": str(gdf.crs),
        "target_crs": str(reprojected.crs),
        "features_count": len(reprojected),
    })

# -- --
@tool_registry.register(
    name="generate_random_points",
    description="Generate random points within a polygon dataset.",
    parameters={
        "dataset_name": {"type": "string", "description": "Polygon dataset to generate points within"},
        "num_points": {"type": "integer", "description": "Number of random points to generate"},
        "output_name": {"type": "string", "description": "Name for output point dataset"},
    },
    returns="Generated points count",
)
def generate_random_points(dataset_name: str, num_points: int, output_name: str):
    """"""
    from shapely.geometry import Point
    from shapely.ops import unary_union

    gdf = _loaded_datasets.get(dataset_name)
    if gdf is None:
        return json.dumps({"error": f"Dataset '{dataset_name}' not found"})

    boundary = unary_union(gdf.geometry)
    minx, miny, maxx, maxy = boundary.bounds
    points = []
    attempts = 0
    while len(points) < num_points and attempts < num_points * 100:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        p = Point(x, y)
        if boundary.contains(p):
            points.append(p)
        attempts += 1

    result = gpd.GeoDataFrame(
        {"id": range(len(points))},
        geometry=points,
        crs=gdf.crs
    )
    _loaded_datasets[output_name] = result

    return json.dumps({
        "dataset_name": output_name,
        "points_generated": len(points),
        "requested": num_points,
    })

# -- --
@tool_registry.register(
    name="create_grid",
    description="Create a regular grid of polygons covering a dataset's extent.",
    parameters={
        "dataset_name": {"type": "string", "description": "Reference dataset for extent"},
        "cell_size": {"type": "number", "description": "Grid cell size in CRS units"},
        "grid_type": {"type": "string", "description": "Grid type: 'rectangle' or 'hexagon'", "optional": True},
        "output_name": {"type": "string", "description": "Name for output grid dataset"},
    },
    returns="Grid cells count",
)
def create_grid(dataset_name: str, cell_size: float, output_name: str, grid_type: str = "rectangle"):
    """"""
    from shapely.geometry import box

    gdf = _loaded_datasets.get(dataset_name)
    if gdf is None:
        return json.dumps({"error": f"Dataset '{dataset_name}' not found"})

    minx, miny, maxx, maxy = gdf.total_bounds
    cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            cells.append(box(x, y, x + cell_size, y + cell_size))
            y += cell_size
        x += cell_size

    grid = gpd.GeoDataFrame(
        {"cell_id": range(len(cells))},
        geometry=cells,
        crs=gdf.crs
    )
    _loaded_datasets[output_name] = grid

    return json.dumps({
        "dataset_name": output_name,
        "cells": len(cells),
        "cell_size": cell_size,
        "extent": [round(minx, 4), round(miny, 4), round(maxx, 4), round(maxy, 4)],
    })

# -- () --
@tool_registry.register(
    name="join_by_attribute",
    description="Join two datasets by a common attribute column (like SQL JOIN).",
    parameters={
        "left_dataset": {"type": "string", "description": "Left dataset name"},
        "right_dataset": {"type": "string", "description": "Right dataset name"},
        "left_column": {"type": "string", "description": "Join column in left dataset"},
        "right_column": {"type": "string", "description": "Join column in right dataset"},
        "how": {"type": "string", "description": "Join type: 'left', 'right', 'inner', 'outer'", "optional": True},
        "output_name": {"type": "string", "description": "Name for joined dataset"},
    },
    returns="Joined dataset info",
)
def join_by_attribute(left_dataset: str, right_dataset: str,
                      left_column: str, right_column: str,
                      output_name: str, how: str = "left"):
    """"""
    left = _loaded_datasets.get(left_dataset)
    right = _loaded_datasets.get(right_dataset)
    if left is None:
        return json.dumps({"error": f"Dataset '{left_dataset}' not found"})
    if right is None:
        return json.dumps({"error": f"Dataset '{right_dataset}' not found"})

    # right GeoDataFrame geometry 
    right_df = right.drop(columns=['geometry'], errors='ignore')
    result = left.merge(right_df, left_on=left_column, right_on=right_column, how=how)
    if not isinstance(result, gpd.GeoDataFrame):
        result = gpd.GeoDataFrame(result, geometry='geometry', crs=left.crs)

    _loaded_datasets[output_name] = result

    return json.dumps({
        "dataset_name": output_name,
        "features_count": len(result),
        "columns": list(result.columns)[:15],
        "join_type": how,
    })

# -- CSV/Excel --
@tool_registry.register(
    name="export_to_csv",
    description="Export a dataset's attribute table to CSV file (without geometry).",
    parameters={
        "dataset_name": {"type": "string", "description": "Dataset to export"},
        "output_file": {"type": "string", "description": "Output CSV file path"},
        "include_coords": {"type": "boolean", "description": "Include X,Y coordinate columns", "optional": True},
    },
    returns="Export info",
)
def export_to_csv(dataset_name: str, output_file: str, include_coords: bool = False):
    """ CSV"""
    gdf = _loaded_datasets.get(dataset_name)
    if gdf is None:
        return json.dumps({"error": f"Dataset '{dataset_name}' not found"})

    df = gdf.copy()
    if include_coords and hasattr(gdf, 'geometry'):
        df['longitude'] = gdf.geometry.x if gdf.geometry.geom_type.iloc[0] == 'Point' else gdf.geometry.centroid.x
        df['latitude'] = gdf.geometry.y if gdf.geometry.geom_type.iloc[0] == 'Point' else gdf.geometry.centroid.y

    df_out = df.drop(columns=['geometry'], errors='ignore')
    _ensure_dir(output_file)
    df_out.to_csv(output_file, index=False)

    return json.dumps({
        "output_file": output_file,
        "rows": len(df_out),
        "columns": list(df_out.columns)[:15],
    })
