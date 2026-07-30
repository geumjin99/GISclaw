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

# 2) Thin pip layer: web server, cloud SDKs, analysis libs
COPY app/requirements.txt /app/app/requirements.txt
RUN pip install --no-cache-dir -r /app/app/requirements.txt

# 3) App code (src/ is the agent core, app/ is the product server+web)
COPY src/ /app/src/
COPY app/ /app/app/

# 4) Runtime config
ENV GISCLAW_WORKSPACE=/workspace \
    PYTHONUNBUFFERED=1
RUN mkdir -p /workspace
EXPOSE 8765

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8765"]
