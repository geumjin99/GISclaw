# Example — Urban heat island, Madison (Wisconsin)

Two layers that let you exercise GISclaw end to end in a couple of minutes.

| File | Geometry | Features | CRS | Key fields |
|---|---|---|---|---|
| `Temperature.geojson` | Point | 139 | EPSG:6610 | `TemperatureF` (73.43 – 83.87 °F) |
| `block.geojson` | Polygon | 269 | EPSG:32618 | `Block_Groups_TOTPOP10`, `Block_Groups_PopOver65`, … |

The two layers deliberately use **different coordinate reference systems** —
that is the most common real-world trap, and a good test of whether the agent
handles reprojection before joining.

## Attribution — please read

These files come from **GeoAnalystBench** (task 1), released by GeoDS under the
**Apache License 2.0**:

> Zhang et al. (2025). *GeoAnalystBench: A GeoAI benchmark for assessing large
> language models for spatial analysis workflow and code generation.*
> Transactions in GIS. <https://github.com/GeoDS/GeoAnalystBench>

They are redistributed here **unmodified in content**; only their directory
location changed. A copy of the Apache-2.0 licence is in this folder
(`LICENSE-Apache-2.0.txt`), as that licence requires.

**If you publish work using this data, cite Zhang et al. (2025).** GISclaw's own
paper is a separate citation — see the repository README.

## Try it

1. Copy this folder into the workspace GISclaw mounts (`./projects/`).
2. In the app: **Project → ＋ New project…**, then **Project → ＋ Add data…**
   and attach both files.
3. Paste this prompt:

```
Interpolate the TemperatureF point measurements into a continuous surface,
then compute the zonal mean temperature for each census block.

Save to pred_results/:
  1. blocks_meantemp.geojson  — blocks with a new field mean_temp_f
  2. heat_choropleth.png      — a choropleth of mean_temp_f

Report the 5 hottest blocks and how many blocks ended up with no value.
```

Expect the agent to notice the CRS mismatch on its own, reproject, and report
how many of the 269 blocks it could not compute a value for. A run that claims
all 269 have values deserves a second look.
