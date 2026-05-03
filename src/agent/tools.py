"""
GIS Toolkit — Standard tool set callable by the Agent

Design:
- load_vector / load_raster auto-inject into Sandbox namespace
- Return unified schema summaries
- All tools return plain-text Observations
"""
import os
from typing import Dict, Any, Optional

class GISToolkit:
    """GIS analysis toolkit, bound to a PythonSandbox instance"""
    
    def __init__(self, sandbox, data_dir: str = "dataset"):
        self.sandbox = sandbox
        self.data_dir = data_dir
        
        # Tool registry
        self.tools = {
            "list_files": self.list_files,
            "load_vector": self.load_vector,
            "load_raster": self.load_raster,
            "execute": self.execute_code,
            "inspect": self.inspect_var,
            "finish": self.finish,
        }
        
        # Tool descriptions (for System Prompt)
        self.tool_descriptions = """1. list_files()
   List all data files in dataset/ directory with their type and size. No arguments needed.

2. load_vector(path, var_name)
   Load vector data (shp/geojson/gpkg), store into variable var_name.
   Returns: row count, column names, CRS, first 3 rows sample, unique values for key columns.

3. load_raster(path, var_name)
   Load raster data (tif), store into variable var_name.
   Returns: dimensions, band count, CRS, data type, value range.

4. execute(code)
   Execute a code snippet in a persistent Python environment. Variables persist across calls.
   Returns: stdout + stderr + new variable summary.
   Note: Write only 3-10 lines of code per call, print intermediate results to verify.

5. inspect(var_name)
   Check variable state: type, shape, first few rows, statistical summary.

6. finish(summary)
   Mark task as complete. Provide a result summary."""
    
    def run(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a tool, return Observation text"""
        if tool_name not in self.tools:
            return f"[fail] Unknown tool '{tool_name}'. Available tools: {', '.join(self.tools.keys())}"
        
        try:
            return self.tools[tool_name](**args)
        except TypeError as e:
            return f"[fail] Tool '{tool_name}' argument error: {e}\nPlease check parameter names and types."
        except Exception as e:
            return f"[fail] Tool execution error: {type(e).__name__}: {e}"
    
    # -- Tool implementations ----------------------------------------
    
    def list_files(self) -> str:
        """List all files in the data directory"""
        data_path = os.path.join(self.sandbox.work_dir, self.data_dir)
        if not os.path.exists(data_path):
            return f"[fail] Data directory does not exist: {self.data_dir}/"
        
        files = sorted(os.listdir(data_path))
        if not files:
            return f"Data directory {self.data_dir}/ is empty"
        
        lines = [f"[path] {self.data_dir}/ directory ({len(files)} files):"]
        for fname in files:
            fpath = os.path.join(data_path, fname)
            size = os.path.getsize(fpath)
            ext = os.path.splitext(fname)[1].lower()
            
            # File type labels
            type_map = {
                '.geojson': 'Vector', '.shp': 'Vector', '.gpkg': 'Vector',
                '.tif': 'Raster', '.tiff': 'Raster',
                '.csv': 'Table', '.xlsx': 'Table', '.xls': 'Table',
                '.nc': 'NetCDF', '.nc4': 'NetCDF',
                '.lyrx': '[warn] ArcGIS-only (unreadable)',
                '.json': 'JSON', '.txt': 'Text',
            }
            ftype = type_map.get(ext, 'Other')
            
            if size > 1024 * 1024:
                size_str = f"{size/1024/1024:.1f} MB"
            else:
                size_str = f"{size/1024:.1f} KB"
            
            lines.append(f"  - {fname} [{ftype}, {size_str}]")
        
        return "\n".join(lines)
    
    def load_vector(self, path: str, var_name: str) -> str:
        """Load vector data and inject into sandbox namespace"""
        # Generate loading code
        code = f"""{var_name} = gpd.read_file('{path}')
print(f"Rows: {{len({var_name})}}")
print(f"CRS: {{{var_name}.crs}}")
print(f"Columns: {{list({var_name}.columns)}}")
print()
# Show first 3 rows
print({var_name}.head(3).to_string())
print()
# Show unique values for key columns
for col in [c for c in {var_name}.columns if c != 'geometry'][:6]:
    try:
        uniques = {var_name}[col].dropna().unique()
        if len(uniques) <= 10:
            print(f"{{col}} unique values: {{list(uniques)}}")
        else:
            print(f"{{col}}: {{len(uniques)}} unique values, e.g.: {{list(uniques[:5])}}")
    except:
        pass
"""
        result = self.sandbox.execute(code)
        return result.to_observation()
    
    def load_raster(self, path: str, var_name: str) -> str:
        """Load raster data and inject into sandbox namespace
        
        Injects two variables:
        - {var_name}: ndarray — single band: (H, W), multi-band: (bands, H, W)
        - {var_name}_meta: dict (contains crs, transform, nodata, width, height, dtype, count)
        """
        code = f"""_src = rasterio.open('{path}')
if _src.count > 1:
    {var_name} = _src.read().astype(float)
else:
    {var_name} = _src.read(1).astype(float)
{var_name}_meta = dict(
    crs=_src.crs,
    transform=_src.transform,
    width=_src.width,
    height=_src.height,
    nodata=_src.nodata,
    dtype=_src.dtypes[0],
    count=_src.count,
    bounds=_src.bounds,
)
print(f"Size: {{_src.width}}x{{_src.height}}")
print(f"Bands: {{_src.count}}")
print(f"CRS: {{_src.crs}}")
print(f"Dtype: {{_src.dtypes[0]}}")
if _src.count > 1:
    print(f"Shape: {{{var_name}.shape}} (bands, height, width)")
    for _b in range(_src.count):
        _band = {var_name}[_b]
        print(f"  Band {{_b+1}}: min={{_band[~__import__('numpy').isnan(_band)].min():.4g}}, max={{_band[~__import__('numpy').isnan(_band)].max():.4g}}" if not __import__('numpy').all(__import__('numpy').isnan(_band)) else f"  Band {{_b+1}}: all NaN")
    print(f"\\nHint: {var_name} has {{_src.count}} bands. Access bands by index: {var_name}[0]=Band1, {var_name}[1]=Band2, etc.")
else:
    print(f"Value range: min={{{var_name}.min():.4g}}, max={{{var_name}.max():.4g}}")
print(f"Bounds: {{_src.bounds}}")
if _src.nodata is not None:
    print(f"NoData: {{_src.nodata}}")
    import numpy as _np
    {var_name}[{var_name} == _src.nodata] = _np.nan
    if _src.count == 1:
        print(f"NoData replaced with NaN, valid range: min={{{var_name}[~_np.isnan({var_name})].min():.4g}}, max={{{var_name}[~_np.isnan({var_name})].max():.4g}}")
    else:
        print(f"NoData replaced with NaN across all bands")
print(f"Transform: {{_src.transform}}")
if _src.count == 1:
    print(f"\\\\nHint: {var_name} is ndarray(float) shape (H, W), {var_name}_meta is dict(crs/transform/nodata etc.)")
    print(f"To write GeoTIFF: rasterio.open(path, 'w', driver='GTiff', height={var_name}.shape[0], width={var_name}.shape[1], count=1, dtype='float32', crs={var_name}_meta['crs'], transform={var_name}_meta['transform'])")
_src.close()
"""
        result = self.sandbox.execute(code)
        return result.to_observation()
    
    def execute_code(self, code: str) -> str:
        """Execute arbitrary code in sandbox, auto-QC output files after execution"""
        # Record pre-execution pred_results file state
        pred_dir = os.path.join(self.sandbox.work_dir, "pred_results")
        before_state = {}
        if os.path.exists(pred_dir):
            for f in os.listdir(pred_dir):
                fp = os.path.join(pred_dir, f)
                before_state[f] = os.path.getmtime(fp)
        
        result = self.sandbox.execute(code)
        obs = result.to_observation()
        
        # QC: Check newly written/modified output files
        warnings = []
        if os.path.exists(pred_dir):
            for f in os.listdir(pred_dir):
                fp = os.path.join(pred_dir, f)
                mtime = os.path.getmtime(fp)
                # Only check newly created or modified files
                if f not in before_state or mtime > before_state.get(f, 0):
                    size_kb = os.path.getsize(fp) / 1024
                    ext = os.path.splitext(f)[1].lower()
                    
                    # PNG QC: < 30KB may be blank
                    if ext == '.png' and size_kb < 30:
                        warnings.append(
                            f"[warn] Output QC warning: {f} is only {size_kb:.1f}KB, "
                            f"the image may be blank! Use print(len(ax.get_children())) "
                            f"or print(fig.get_axes()[0].has_data()) to check if data is bound to the plot. "
                            f"If blank, re-check that data variables are correct and plot calls are valid."
                        )
                    
                    # TIF QC: Check if all nodata (rough check via file size)
                    if ext == '.tif' and size_kb < 1:
                        warnings.append(
                            f"[warn] Output QC warning: {f} is only {size_kb:.1f}KB, "
                            f"the raster file may be empty! Please check the write logic."
                        )
        
        if warnings:
            obs += "\n\n" + "\n".join(warnings)
        
        return obs
    
    def inspect_var(self, var_name: str) -> str:
        """Check variable state"""
        return self.sandbox.inspect(var_name)
    
    def finish(self, summary: str = "") -> str:
        """Mark task as complete"""
        # Collect output files
        pred_dir = os.path.join(self.sandbox.work_dir, "pred_results")
        output_files = []
        if os.path.exists(pred_dir):
            for f in sorted(os.listdir(pred_dir)):
                fpath = os.path.join(pred_dir, f)
                size = os.path.getsize(fpath) / 1024
                output_files.append(f"{f} ({size:.1f} KB)")
        
        result = f"[list] Task complete\n"
        if summary:
            result += f"Summary: {summary}\n"
        if output_files:
            result += f"Output files ({len(output_files)}):\n"
            for f in output_files:
                result += f"  - {f}\n"
        else:
            result += "[warn] No output files generated! Please ensure results are saved to pred_results/\n"
        
        return result
