"""
Cross-task error memory: fixes learned on earlier tasks are reused on later tasks
"""

class ErrorMemory:
    """-> """

    def __init__(self):
        self._memory = {}
        self._memory.update({
            "has no attribute 'read'": "load_raster ndarray DatasetReader .read() rasterio.open() ",
            "geoplot": "Use geoplot with matplotlib backend (Agg). If projection errors occur, try gcrs.PlateCarree() instead of AlbersEqualArea().",
            "has no attribute 'crs'": "unary_union Shapely Geometry crs gpd.GeoDataFrame(geometry=[...], crs=gdf.crs) ",
            "MemoryError": ": gdf_sample = gdf.sample(n=10000, random_state=42)",
            "ForwardCompatibility": "NVIDIA driver ",
        })

    def lookup(self, error_msg: str) -> str:
        """"""
        for pattern, fix in self._memory.items():
            if pattern.lower() in error_msg.lower():
                return fix
        return ""

    def record(self, error_pattern: str, fix_suggestion: str):
        """->"""
        key = error_pattern.strip()
        if len(key) > 10:
            self._memory[key] = fix_suggestion

    def get_all(self) -> dict:
        return dict(self._memory)

    def __len__(self):
        return len(self._memory)
