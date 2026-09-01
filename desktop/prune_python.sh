#!/usr/bin/env bash
# Remove the parts of a bundled Python that GISclaw never uses: the interpreter's
# own test suite, IDLE, Tk, the installer tooling, and compiled caches (rebuilt
# on first use where the folder is writable, otherwise not needed).
#   bash desktop/prune_python.sh <python-root>
set -euo pipefail
ROOT="$1"
LIB=$(ls -d "$ROOT"/lib/python3.* | head -1)
for d in test idlelib tkinter turtledemo ensurepip lib2to3 \
         site-packages/pip site-packages/setuptools site-packages/selenium site-packages/pkg_resources; do
  rm -rf "$LIB/$d"
done
rm -f "$LIB"/lib-dynload/_tkinter*.so "$ROOT"/lib/libtcl* "$ROOT"/lib/libtk* 2>/dev/null || true
rm -rf "$ROOT"/lib/tcl* "$ROOT"/lib/tk* "$ROOT"/lib/itcl* "$ROOT"/share 2>/dev/null || true
find "$ROOT" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$ROOT" -name "*.pyc" -delete
du -sh "$ROOT" | awk '{print "python after trim: "$1}'
