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
Python Sandbox — Interactive execution environment

Persistent namespace, Jupyter-like. Variables persist throughout task execution.
Supports incremental code snippet execution, variable inspection, and full code traceback.

Design:
- stdout truncated from the front, stderr truncated from the tail (to see actual errors)
- Pre-imports common GIS libraries to reduce model's import burden
"""
import ctypes
import io
import os
import sys
import threading
import time
import traceback
from typing import Callable, Dict, Optional


class _StreamRouter:
    """A stand-in for sys.stdout / sys.stderr that delivers per thread.

    `contextlib.redirect_stdout` swaps the stream for the whole process, so
    while a snippet ran, anything any other thread printed landed in the
    snippet's output — and the snippet's output could leak the other way. The
    router sends a thread's writes to the buffer bound to it, and everything
    else to the real stream.
    """

    def __init__(self, fallback):
        self._fallback = fallback
        self._targets = {}

    def bind(self, tid, stream):
        self._targets[tid] = stream

    def unbind(self, tid):
        self._targets.pop(tid, None)

    def _current(self):
        return self._targets.get(threading.get_ident(), self._fallback)

    def write(self, s):
        return self._current().write(s)

    def writelines(self, lines):
        return self._current().writelines(lines)

    def flush(self):
        try:
            return self._current().flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self._fallback, name)


def _routers():
    """Install the routers once, on first use; return (stdout, stderr)."""
    if not isinstance(sys.stdout, _StreamRouter):
        sys.stdout = _StreamRouter(sys.stdout)
    if not isinstance(sys.stderr, _StreamRouter):
        sys.stderr = _StreamRouter(sys.stderr)
    return sys.stdout, sys.stderr


class SandboxInterrupt(BaseException):
    """Raised inside the running snippet when it is cancelled or times out.

    A BaseException, not an Exception, so a snippet's own `except Exception`
    cannot swallow it.
    """


def _raise_in_thread(thread: threading.Thread, exc_type) -> bool:
    """Ask the interpreter to raise `exc_type` in `thread` at its next bytecode.

    This is the only way to stop code that is already running in another
    thread. It takes effect when that thread returns to Python; a snippet deep
    inside a long C call (a big NumPy operation, a GDAL read) is interrupted
    only when that call returns.
    """
    tid = thread.ident
    if tid is None:
        return False
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(tid), ctypes.py_object(exc_type))
    if res > 1:
        # More than one thread state matched — undo, never shotgun the interpreter.
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)
        return False
    return res == 1


class ExecutionResult:
    """Result of a single code execution"""
    def __init__(self, stdout: str = "", stderr: str = "",
                 success: bool = True, new_vars: Dict[str, str] = None,
                 execution_time: float = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.success = success
        self.new_vars = new_vars or {}  # {var_name: type_summary}
        self.execution_time = execution_time
    
    def to_observation(self, max_stdout=800, max_stderr=600) -> str:
        """Convert to Agent-readable Observation text
        
        Key: stdout truncated from front, stderr from tail (stack trace tail has the real error)
        """
        parts = []
        
        if self.success:
            parts.append("✅ Execution successful")
        else:
            parts.append("❌ Execution failed")
        
        if self.stdout.strip():
            stdout_display = self.stdout.strip()
            if len(stdout_display) > max_stdout:
                stdout_display = stdout_display[:max_stdout] + "\n... (truncated)"
            parts.append(f"stdout:\n{stdout_display}")
        
        if self.stderr.strip():
            stderr_display = self.stderr.strip()
            if len(stderr_display) > max_stderr:
                # Key: truncate stderr from the end, keeping the real error message
                stderr_display = "... (stack trace omitted)\n" + stderr_display[-max_stderr:]
            parts.append(f"stderr:\n{stderr_display}")
        
        if self.new_vars:
            var_lines = [f"  {k}: {v}" for k, v in self.new_vars.items()]
            parts.append("Variable changes:\n" + "\n".join(var_lines))
        
        return "\n".join(parts)




class PythonSandbox:
    """Interactive Python execution sandbox"""
    
    def __init__(self, work_dir: str = ".", timeout: int = 60,
                 should_stop: Optional[Callable[[], bool]] = None):
        """
        should_stop: optional callable polled while a snippet runs. When it
            returns True the snippet is interrupted, the same way a timeout
            interrupts it. This is how a Stop button reaches running code.
        """
        self.work_dir = os.path.abspath(work_dir)
        self.timeout = timeout
        self.should_stop = should_stop
        self.namespace = {"__builtins__": __builtins__}
        self.code_history = []  # List of successfully executed code snippets
        self._known_vars = set()  # Track existing variables
        # A snippet that ignored its interrupt is still running somewhere in
        # this process. Nothing further can safely share the namespace with it.
        self.stuck = False
        self._init_environment()
    
    def _init_environment(self):
        """Pre-import common libraries + set working directory"""
        init_code = """
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, calculate_default_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.close('all')
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')
os.makedirs('pred_results', exist_ok=True)
"""
        # The working directory is process-wide, so it is borrowed for the
        # duration of a snippet and handed back — never left pointing here.
        original_dir = os.getcwd()
        os.chdir(self.work_dir)
        try:
            exec(init_code, self.namespace)
            self._known_vars = set(self.namespace.keys())
        except Exception as e:
            print(f"Sandbox init warning: {e}")
        finally:
            os.chdir(original_dir)
    
    def execute(self, code: str) -> ExecutionResult:
        """Execute code snippet in persistent namespace"""

        # Import blocker: intercept unavailable packages before execution
        BLOCKED_PACKAGES = {
            'arcpy': 'Use geopandas + rasterio + shapely instead.',
            'pykrige': 'Use scipy.interpolate.griddata or scipy.interpolate.Rbf instead.',
            'skimage': 'Use scipy.ndimage or numpy for image processing.',
        }
        for line in code.split('\n'):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            for pkg, alt in BLOCKED_PACKAGES.items():
                if f'import {pkg}' in stripped or f'from {pkg}' in stripped:
                    return ExecutionResult(
                        stdout="",
                        stderr=f"❌ Package '{pkg}' is NOT available in this environment.\n"
                               f"Alternative: {alt}\n"
                               f"Do NOT attempt to import {pkg} again — use the alternative packages.",
                        success=False,
                        new_vars={},
                        execution_time=0.0,
                    )
        
        if self.stuck:
            return ExecutionResult(
                stdout="", success=False, new_vars={}, execution_time=0.0,
                stderr="❌ The previous snippet could not be interrupted and is "
                       "still running. This session cannot execute more code.")

        # Save current variable snapshot
        vars_before = set(self.namespace.keys())

        # Capture stdout/stderr
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        t0 = time.time()
        success = True
        interrupted = ""

        # The snippet runs in its own thread so that this one can watch the
        # clock and the stop flag. Without that, a `while True` written by the
        # model would hang the whole process with no way back.
        original_dir = os.getcwd()
        os.chdir(self.work_dir)
        namespace = self.namespace

        out_router, err_router = _routers()

        def _target():
            nonlocal success
            tid = threading.get_ident()
            out_router.bind(tid, stdout_buf)
            err_router.bind(tid, stderr_buf)
            try:
                exec(code, namespace)
            except SandboxInterrupt:
                success = False
            except BaseException:
                success = False
                stderr_buf.write(traceback.format_exc())
            finally:
                out_router.unbind(tid)
                err_router.unbind(tid)

        worker = threading.Thread(target=_target, name="sandbox-exec", daemon=True)
        try:
            if True:
                worker.start()
                while worker.is_alive():
                    worker.join(0.2)
                    if not worker.is_alive():
                        break
                    elapsed = time.time() - t0
                    if self.should_stop and self.should_stop():
                        interrupted = "⏹ Execution stopped by the user."
                    elif elapsed > self.timeout:
                        interrupted = (f"⏰ Execution took longer than the "
                                       f"{self.timeout}s limit and was stopped.")
                    if interrupted:
                        _raise_in_thread(worker, SandboxInterrupt)
                        # Give it a moment to unwind; a snippet stuck inside a
                        # C call will not, and then the session is over.
                        worker.join(5.0)
                        if worker.is_alive():
                            self.stuck = True
                            interrupted += (" The code did not respond to the "
                                            "interrupt; this session is closed.")
                        success = False
                        break
        finally:
            os.chdir(original_dir)

        if interrupted:
            stderr_buf.write("\n" + interrupted)

        exec_time = time.time() - t0
        
        # Detect new/modified variables
        vars_after = set(self.namespace.keys())
        new_var_names = vars_after - vars_before
        new_vars = {}
        for name in new_var_names:
            if name.startswith('_'):
                continue
            new_vars[name] = self._describe_var(name)
        
        result = ExecutionResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            success=success,
            new_vars=new_vars,
            execution_time=exec_time,
        )
        
        # Record code on success
        if success:
            self.code_history.append(code)
            self._known_vars = vars_after
        
        return result
    
    def inspect(self, var_name: str) -> str:
        """Inspect variable state, return human-readable description"""
        if var_name not in self.namespace:
            return f"❌ Variable '{var_name}' does not exist. Available variables: {self._list_user_vars()}"
        
        return self._describe_var(var_name, detailed=True)
    
    def _describe_var(self, var_name: str, detailed: bool = False) -> str:
        """Generate human-readable description of a variable"""
        obj = self.namespace.get(var_name)
        if obj is None:
            return "None"
        
        try:
            import geopandas as gpd
            import pandas as pd
            import numpy as np
            
            if isinstance(obj, gpd.GeoDataFrame):
                desc = f"GeoDataFrame, {len(obj)} rows, CRS={obj.crs}"
                cols = [c for c in obj.columns if c != 'geometry']
                desc += f"\n  Columns: {', '.join(cols[:15])}"
                if detailed and len(obj) > 0:
                    desc += f"\n  First 3 rows:\n{obj.head(3).to_string()}"
                    # Show unique values for key columns
                    for col in cols[:5]:
                        try:
                            uniques = obj[col].dropna().unique()
                            if len(uniques) <= 8:
                                desc += f"\n  {col} unique values: {list(uniques[:8])}"
                            else:
                                desc += f"\n  {col}: {len(uniques)} unique values, e.g.: {list(uniques[:3])}"
                        except:
                            pass
                return desc
            
            elif isinstance(obj, pd.DataFrame):
                desc = f"DataFrame, {len(obj)} rows, {len(obj.columns)} cols"
                desc += f"\n  Columns: {', '.join(obj.columns[:15])}"
                if detailed:
                    desc += f"\n  First 3 rows:\n{obj.head(3).to_string()}"
                return desc
            
            elif isinstance(obj, np.ndarray):
                desc = f"ndarray, shape={obj.shape}, dtype={obj.dtype}"
                if obj.size > 0:
                    desc += f", min={obj.min():.4g}, max={obj.max():.4g}"
                    if np.isnan(obj).any():
                        desc += f", NaN ratio={np.isnan(obj).mean():.1%}"
                return desc
            
            elif isinstance(obj, (gpd.GeoSeries,)):
                desc = f"GeoSeries, {len(obj)} geometries"
                if obj.crs:
                    desc += f", CRS={obj.crs}"
                return desc
            
            elif hasattr(obj, 'read'):  # rasterio DatasetReader
                try:
                    desc = f"Raster, {obj.width}x{obj.height}, {obj.count} band(s)"
                    desc += f", CRS={obj.crs}, dtype={obj.dtypes[0]}"
                    if obj.nodata is not None:
                        desc += f", nodata={obj.nodata}"
                    return desc
                except:
                    pass
            
            # Other types
            s = str(obj)
            if len(s) > 200:
                s = s[:200] + "..."
            return f"{type(obj).__name__}: {s}"
        
        except Exception as e:
            return f"{type(obj).__name__} (description error: {e})"
    
    def _list_user_vars(self) -> str:
        """List user-created variables (excluding builtins and underscore-prefixed)"""
        skip = {'__builtins__', 'os', 'np', 'pd', 'gpd', 'rasterio', 
                'matplotlib', 'plt', 'Point', 'Polygon', 'MultiPolygon',
                'box', 'unary_union', 'warnings', 'reproject',
                'calculate_default_transform'}
        user_vars = [k for k in self.namespace.keys() 
                     if not k.startswith('_') and k not in skip]
        return ', '.join(user_vars) if user_vars else "(none)"
    
    def get_variables_summary(self) -> str:
        """Public API: list current user variables in sandbox"""
        return self._list_user_vars()
    
    def get_full_code(self) -> str:
        """Return merged full script"""
        header = """import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, calculate_default_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, MultiPolygon, box
from shapely.ops import unary_union
import warnings
warnings.filterwarnings('ignore')
os.makedirs('pred_results', exist_ok=True)
"""
        return header + "\n\n" + "\n\n".join(self.code_history)
    
    def reset(self):
        """Reset sandbox state"""
        self.namespace = {"__builtins__": __builtins__}
        self.code_history = []
        self._known_vars = set()
        self._init_environment()
