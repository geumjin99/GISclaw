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

"""GISclaw product — logging helpers.

Two log surfaces (see plan): a process-level app log, and a per-run log +
structured trace.jsonl written inside each analysis run folder. Kept small on
purpose — this is the "proper software has logs" layer, nothing enterprise.
"""
import json
import logging
import os
from datetime import datetime


def get_app_logger(log_path: str) -> logging.Logger:
    """Process-level logger, writes to app/server.log and stderr."""
    logger = logging.getLogger("gisclaw")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


class RunRecorder:
    """Per-run provenance: a run.log line-logger plus a trace.jsonl event stream.

    One RunRecorder is created per analysis run. Everything the agent emits
    (thought / action / observation / result / done) is appended to trace.jsonl,
    so a finished run folder is a complete, replayable audit trail.
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.trace_path = os.path.join(run_dir, "trace.jsonl")
        self.log_path = os.path.join(run_dir, "run.log")
        self._log_fh = open(self.log_path, "a", encoding="utf-8")
        self._trace_fh = open(self.trace_path, "a", encoding="utf-8")

    def log(self, msg: str):
        self._log_fh.write(f"{datetime.now().isoformat(timespec='seconds')}  {msg}\n")
        self._log_fh.flush()

    def event(self, ev: dict):
        """Append one structured event (a dict) as a JSON line."""
        self._trace_fh.write(json.dumps(ev, ensure_ascii=False) + "\n")
        self._trace_fh.flush()

    def close(self):
        for fh in (self._log_fh, self._trace_fh):
            try:
                fh.close()
            except Exception:
                pass
