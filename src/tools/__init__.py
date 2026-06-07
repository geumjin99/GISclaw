# Vendored for IG4.0: each submodule import is wrapped in try/except so a
# missing optional dependency (rasterio, matplotlib, etc.) does not abort
# loading of the entire toolkit. Failures are exposed via _GISCLAW_TOOLS_FAILED.
_GISCLAW_TOOLS_FAILED: dict = {}
for _name in (
    "vector_tools",
    "analysis_tools",
    "advanced_tools",
    "raster_tools",
    "conversion_tools",
    "terrain_tools",
    "viz_tools",
):
    try:
        __import__(f"{__name__}.{_name}", fromlist=["*"])
    except Exception as _exc:  # noqa: BLE001
        _GISCLAW_TOOLS_FAILED[_name] = f"{type(_exc).__name__}: {_exc}"
del _name
