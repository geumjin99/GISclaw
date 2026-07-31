# Third-party notices

GISclaw is licensed under the **GNU Affero General Public License v3.0 or
later** (see `LICENSE`); copyright (C) 2026 Han Jinzhen (see `COPYRIGHT`). It
bundles or depends on the third-party material listed below, each under its own
licence.

**Licence compatibility.** Every bundled component is under a permissive licence
that is one-way compatible with the AGPLv3 — Apache-2.0 (per GPLv3 §7 and the
FSF's compatibility list), BSD-2-Clause, and ISC. Combining them into this
AGPL-licensed work is permitted; their own notices and licence texts are
preserved as required, and each remains under its original terms when taken
separately.

---

## 1. Example dataset — GeoAnalystBench (Apache License 2.0)

**What is bundled:** `examples/urban-heat-madison/Temperature.geojson` and
`examples/urban-heat-madison/block.geojson`.

**Origin:** these two files are taken from **GeoAnalystBench**, task 1
("Urban heat island and at-risk elderly population, Madison, Wisconsin").

- Repository: <https://github.com/GeoDS/GeoAnalystBench>
- Licence: **Apache License 2.0**
- Citation: Zhang et al. (2025), *GeoAnalystBench: A GeoAI benchmark for
  assessing large language models for spatial analysis workflow and code
  generation*, **Transactions in GIS**.

**Statement of changes** (required by Apache-2.0 §4(b)): the two GeoJSON files
are redistributed here **unmodified in content**. They have been moved out of
the benchmark's directory layout into `examples/urban-heat-madison/`, and are
accompanied by a README describing how to load them into GISclaw. No attribute
values, geometries, or coordinate reference systems were altered. **No other
part of GeoAnalystBench is redistributed** — the benchmark's reference scripts,
gold-standard outputs, task definitions, and the remaining 49 tasks are not
included in this repository.

**A copy of the Apache License 2.0** is provided at
`examples/urban-heat-madison/LICENSE-Apache-2.0.txt`, together with the
attribution notice, as required by §4(a) and §4(d).

**If you use this dataset in published work, cite Zhang et al. (2025), not us.**

---

## 2. Map tiles and basemap

The map view loads raster tiles from CARTO's public basemap service, which is
derived from **OpenStreetMap** data.

- Map data © OpenStreetMap contributors, licensed under the
  [Open Database Licence (ODbL)](https://www.openstreetmap.org/copyright).
- Tiles © [CARTO](https://carto.com/attributions).

Attribution is displayed in the map's own attribution control at runtime. If
you deploy GISclaw publicly, review CARTO's usage limits and consider pointing
`app/web/app.js` at your own tile source.

---

## 3. Bundled JavaScript

| Component | Version | Licence | Location |
|---|---|---|---|
| [Leaflet](https://leafletjs.com/) | 1.9.4 | BSD-2-Clause | `app/web/vendor/leaflet/` |
| [Lucide](https://lucide.dev/) icons | — | ISC | inlined as SVG paths in `app/web/app.js` |

Leaflet's own licence text ships inside `app/web/vendor/leaflet/leaflet.js`.

---

## 4. Python dependencies

The container installs the open-source geospatial and ML stack from
conda-forge and PyPI — GeoPandas, rasterio, Fiona, Shapely, pyproj, NumPy,
pandas, SciPy, scikit-learn, matplotlib, and others. Each is distributed under
its own licence (predominantly BSD-3-Clause, MIT, or Apache-2.0); see
`app/requirements.txt` and the `Dockerfile` for the full list, and consult each
project for its licence text. GISclaw does not vendor or modify any of them.

---

## 5. Model providers

GISclaw calls hosted large language models (OpenAI, Anthropic, DeepSeek,
Google, or any OpenAI-compatible endpoint you configure). No model weights are
bundled. Your use of those services is governed by each provider's own terms,
and **you supply your own API keys** — none ship with this software.
