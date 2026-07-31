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
What went wrong once, and what fixed it.

Scoped to a single run: when a tool call fails, the loop looks the error up
here and passes the suggestion back with the observation, so the model is not
left rediscovering the same trap. Fixes it finds itself are recorded as it
goes.
"""


class ErrorMemory:
    """A small error-pattern -> suggested-fix table."""

    def __init__(self):
        # Traps seen often enough to be worth stating up front.
        self._memory = {
            "has no attribute 'read'":
                "load_raster gives you an ndarray, not a DatasetReader. Open the "
                "file with rasterio.open() if you need .read().",
            "has no attribute 'crs'":
                "unary_union returns a bare Shapely geometry, which has no CRS. "
                "Wrap it: gpd.GeoDataFrame(geometry=[...], crs=gdf.crs).",
            "MemoryError":
                "Too much data for memory. Work on a sample first: "
                "gdf_sample = gdf.sample(n=10000, random_state=42).",
        }

    def lookup(self, error_msg: str) -> str:
        """Return the suggestion for the first pattern this error matches."""
        for pattern, fix in self._memory.items():
            if pattern.lower() in error_msg.lower():
                return fix
        return ""

    def record(self, error_pattern: str, fix_suggestion: str):
        """Remember a fix that worked, keyed on the error it addressed."""
        key = error_pattern.strip()
        if len(key) > 10:
            self._memory[key] = fix_suggestion

    def get_all(self) -> dict:
        return dict(self._memory)

    def __len__(self):
        return len(self._memory)
