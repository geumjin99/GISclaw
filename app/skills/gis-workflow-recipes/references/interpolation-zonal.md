# Recipe — point measurements → surface → per-polygon summary

The shape: scattered sample points carry a measured value (temperature, PM2.5,
water table, noise). You need a continuous surface and/or a value per reporting
unit (census block, district, parcel).

## Steps

1. **Inspect both layers.** Point count, polygon count, CRS of each, the exact
   name of the value field, and its min/max. Confirm the value field is numeric
   and check the null count.

2. **Unify CRS — to the polygon layer's CRS**, because the polygons are the
   reporting units. Use a projected CRS in metres; interpolation over degrees
   distorts distance weighting.

3. **Decide: do you actually need a raster?**
   - "mean value per polygon" from points that fall inside the polygons →
     a spatial join plus a group-by is enough, and is exact. Prefer it.
   - "continuous surface", "map of the whole area", or polygons with no points
     inside them → you need interpolation.

4. **Interpolate** (only if step 3 says so). Use the `idw` geoprocess op. For a
   smoother surface, `scipy.interpolate.griddata` or `Rbf` in `execute()`;
   `pykrige` is not installed. Set the output resolution deliberately: aim for
   roughly 100–500 cells across the study area extent, not "whatever the
   default is".

5. **Summarise per polygon** with the `zonal_statistics` op against the raster,
   or `spatial_join` + group-by for the exact-points route.

6. **Join back and map.** Attach the summary column to the original polygon
   layer, keeping every polygon — including those with no data, which must stay
   as null rather than silently disappearing or becoming zero.

## Verification — do not skip

- Compare the surface's min/max against the **input points'** min/max. A surface
  may extrapolate slightly beyond the samples; it must not be wildly outside
  them. An unconstrained Kriging run in these experiments once produced
  −27925 °F from inputs in the 73–84 °F range and still "succeeded".
- Print how many polygons received a value and how many are null. State both.
  All-null means the join or the CRS is wrong, not that the data is empty.
- Confirm the mapped field varies. A single-colour choropleth is almost always a
  broken join.

## Common failure

Joining points to polygons with a `within` predicate returns 0 matches when
points sit exactly on boundaries or carry tiny coordinate offsets. If a `within`
join returns 0, check the two layers' bounds overlap, then retry with
`intersects` before assuming the data is wrong.
