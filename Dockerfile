# GISclaw — product image (single-agent ReAct, cloud LLMs).
#
# The GIS stack (GDAL/GEOS/PROJ + geopandas/rasterio/fiona/shapely/pyproj) is the
# painful part to install, so we get it from conda-forge via micromamba instead
# of compiling wheels. Everything else is a thin pip layer on top.
FROM mambaorg/micromamba:1.5-jammy

USER root
WORKDIR /app

# 1) GIS core from conda-forge (handles GDAL cleanly, no compile hell)
RUN micromamba install -y -n base -c conda-forge \
        python=3.11 \
        geopandas rasterio fiona shapely pyproj rtree \
    && micromamba clean --all --yes
ENV PATH=/opt/conda/bin:$PATH

# 2) CJK fonts. Without these, any figure the agent labels in Chinese, Japanese
# or Korean comes out as rows of empty boxes — the base image ships no font with
# CJK glyphs at all.
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 3) Thin pip layer: web server, cloud SDKs, analysis libs
COPY app/requirements.txt /app/app/requirements.txt
RUN pip install --no-cache-dir -r /app/app/requirements.txt

# 4) Make matplotlib pick the CJK font by default, and keep the minus sign
# rendering (the CJK fonts carry no U+2212). Warm the font cache here so the
# first figure of the first run is not delayed by a full font scan.
#
# The JP entry is not a typo and must stay last of the four: the Noto CJK faces
# all ship inside one .ttc collection, and matplotlib sometimes indexes only the
# first face out of it, so 'Noto Sans CJK SC'/'KR' can be absent while 'JP' is
# present. Every face covers Hangul and both Chinese scripts, so JP is a correct
# fallback rather than a Japanese-only one; without it the list drops straight
# to DejaVu Sans and every CJK label renders as an empty box.
ENV MPLCONFIGDIR=/app/.mplconfig
RUN mkdir -p /app/.mplconfig \
    && printf '%s\n' \
       'font.sans-serif: Noto Sans CJK SC, Noto Sans CJK KR, Noto Sans CJK JP, DejaVu Sans, sans-serif' \
       'axes.unicode_minus: False' > /app/.mplconfig/matplotlibrc \
    && python -c "import matplotlib.font_manager"

# 5) App code (src/ is the agent core, app/ is the product server+web)
COPY src/ /app/src/
COPY app/ /app/app/

# 6) Runtime config
ENV GISCLAW_WORKSPACE=/workspace \
    PYTHONUNBUFFERED=1
RUN mkdir -p /workspace
EXPOSE 8765

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8765"]
