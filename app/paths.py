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

"""Where things live on disk, and the rules for getting there safely.

Every path the server hands to the filesystem goes through `safe_join`, which
refuses anything that resolves outside its root. Project folders are named by
`slug`, which keeps non-ASCII names readable while stripping what a filesystem
or shell would object to.
"""
import json
import os
import re
import unicodedata
from datetime import datetime

WORKSPACE = ""

# Per-project record files, surfaced in the UI tree (read-only).
RECORD_FILES = ("JOURNAL.md", "LOG.md", "chat.jsonl")
ARCHIVE_DIR = "_archived"

# Formats that are really a set of files sharing one stem. Attaching only the
# .shp yields a layer nothing can open, and a missing .prj silently drops the
# CRS — so when one member is picked, its siblings travel with it.
SIDECAR_GROUPS = {
    ".shp": [".shp", ".shx", ".dbf", ".prj", ".cpg", ".qpj", ".qix", ".sbn",
             ".sbx", ".fbn", ".fbx", ".ain", ".aih", ".atx", ".shp.xml"],
    ".tab": [".tab", ".dat", ".map", ".id", ".ind"],
    ".mif": [".mif", ".mid"],
}
_SIDECAR_OF = {ext: grp for grp in SIDECAR_GROUPS.values() for ext in grp}


def configure(workspace: str) -> str:
    """Set the workspace root. Called once by the server at start-up."""
    global WORKSPACE
    os.makedirs(workspace, exist_ok=True)
    WORKSPACE = os.path.abspath(workspace)
    return WORKSPACE


def safe_join(root: str, *parts: str) -> str:
    """Join under root and refuse anything that escapes it (path traversal)."""
    root = os.path.abspath(root)
    target = os.path.abspath(os.path.join(root, *parts))
    if target != root and not target.startswith(root + os.sep):
        raise ValueError(f"Path escapes root: {target}")
    return target


def slug(name: str) -> str:
    """A folder name that is safe on every platform but still readable.

    Non-ASCII is kept deliberately: stripping it collapsed a project called
    城市热岛分析 to "project", and the next one collided with it. Only the
    characters a filesystem or shell would object to are replaced. Applying
    this to its own output must not change it — project ids are re-slugged on
    every lookup.
    """
    s = unicodedata.normalize("NFC", name).strip()
    s = re.sub(r'[\x00-\x1f<>:"/\\|?*]+', "_", s)
    s = re.sub(r"\s+", "_", s)
    s = s.strip("._ ")
    return s[:80] or "project"


# ------------------------------------------------------------------ projects --
def project_dir(pid: str) -> str:
    return safe_join(WORKSPACE, slug(pid))


def project_layout(pdir: str) -> None:
    for sub in ("data", "outputs", "runs"):
        os.makedirs(os.path.join(pdir, sub), exist_ok=True)


def read_manifest(pdir: str) -> dict:
    mpath = os.path.join(pdir, "project.json")
    if os.path.exists(mpath):
        try:
            with open(mpath, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def write_manifest(pdir: str, manifest: dict) -> None:
    with open(os.path.join(pdir, "project.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def list_projects() -> list:
    out = []
    for name in sorted(os.listdir(WORKSPACE)):
        pdir = os.path.join(WORKSPACE, name)
        if not os.path.isdir(pdir):
            continue
        if not os.path.exists(os.path.join(pdir, "project.json")):
            continue  # a project is a folder we created with a manifest
        m = read_manifest(pdir)
        out.append({
            "id": name,
            "name": m.get("name", name),
            "created_at": m.get("created_at", ""),
            "notes": m.get("notes", ""),
            "data_count": len(dir_tree(os.path.join(pdir, "data"))),
        })
    return out


def dir_tree(base: str) -> list:
    """Flat list of files under base (relative paths), skipping dotfiles."""
    items = []
    if not os.path.isdir(base):
        return items
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in sorted(files):
            if fn.startswith("."):
                continue
            items.append(os.path.relpath(os.path.join(root, fn), base).replace(os.sep, "/"))
    return sorted(items)


def archive_root() -> str:
    d = os.path.join(WORKSPACE, ARCHIVE_DIR)
    os.makedirs(d, exist_ok=True)
    return d


def new_run_dir(pdir: str, prefix: str = "run") -> str:
    """A fresh, unique folder under runs/ — its name is the run id."""
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    for n in range(100):
        rid = f"{prefix}_{base}" + (f"_{n}" if n else "")
        rdir = os.path.join(pdir, "runs", rid)
        if not os.path.exists(rdir):
            os.makedirs(rdir)
            return rdir
    raise RuntimeError("could not allocate a run folder")


# ---------------------------------------------------------------- sidecars --
def companion_files(src: str) -> list:
    """Every file that has to travel with `src` for it to stay readable."""
    low = src.lower()
    ext = ".shp.xml" if low.endswith(".shp.xml") else os.path.splitext(low)[1]
    group = _SIDECAR_OF.get(ext)
    if not group:
        return [src]
    stem = src[: len(src) - len(ext)]
    out = []
    for e in group:
        for cand in (stem + e, stem + e.upper()):
            if os.path.isfile(cand) and cand not in out:
                out.append(cand)
    return out or [src]
