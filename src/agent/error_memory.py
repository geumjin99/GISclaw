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
跨题错误记忆 — 前面题的修复经验自动传递给后面的题
"""

class ErrorMemory:
    """运行时错误→修复 映射表"""

    def __init__(self):
        self._memory = {}
        # 预置一些已知的常见错误
        self._memory.update({
            "has no attribute 'read'": "load_raster 返回 ndarray 而非 DatasetReader。需要 .read() 时用 rasterio.open() 代替。",
            "geoplot": "Use geoplot with matplotlib backend (Agg). If projection errors occur, try gcrs.PlateCarree() instead of AlbersEqualArea().",
            "has no attribute 'crs'": "unary_union 返回 Shapely Geometry，没有 crs 属性。用 gpd.GeoDataFrame(geometry=[...], crs=gdf.crs) 包装。",
            "MemoryError": "数据量太大导致内存溢出。对输入数据降采样: gdf_sample = gdf.sample(n=10000, random_state=42)",
            "ForwardCompatibility": "NVIDIA driver 问题，忽略此错误。",
        })

    def lookup(self, error_msg: str) -> str:
        """在已知错误中查找匹配的修复建议"""
        for pattern, fix in self._memory.items():
            if pattern.lower() in error_msg.lower():
                return fix
        return ""

    def record(self, error_pattern: str, fix_suggestion: str):
        """记录新的错误→修复映射"""
        # 提取核心错误模式（去掉行号和路径）
        key = error_pattern.strip()
        if len(key) > 10:
            self._memory[key] = fix_suggestion

    def get_all(self) -> dict:
        return dict(self._memory)

    def __len__(self):
        return len(self._memory)
