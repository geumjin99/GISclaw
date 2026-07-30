# Recipe — multi-criteria suitability / risk score

The shape: several layers (slope, distance to road, land cover, population) must
combine into one score — site suitability, landslide risk, service priority.

## Steps

1. **List the criteria explicitly** in your first Thought: for each one, the
   source layer, the direction (higher = better, or higher = worse), and the
   weight. If the task gives weights, use exactly those. If not, use equal
   weights and say so — do not invent a weighting scheme silently.

2. **Bring everything onto one geometry.** Either a common raster grid (same
   CRS, extent and resolution for every layer) or one polygon layer that all
   criteria are joined onto. Mixing the two mid-analysis is where this goes
   wrong.

3. **Derive each criterion.** Typically `slope` from a DEM, distance-to-feature
   from a buffer or proximity computation, class counts from a spatial join.

4. **Normalise before combining.** Raw metres, degrees and counts are not
   comparable. Rescale each criterion to 0–1 (min-max, or an explicit
   reclassification table). **Invert** the ones where higher is worse. State the
   normalisation you used.

5. **Weighted sum**, then classify into the number of classes the task asks for
   (quantile unless told otherwise), and label the classes meaningfully.

6. **Map and tabulate**: the score surface or choropleth, plus a table of area
   or count per class.

## Verification

- Print the min/max of every normalised criterion — each must sit in 0–1. One
  criterion left un-normalised will dominate the whole score.
- Print the final score's range and its distribution across classes. If ~all
  cells land in one class, the normalisation or the weights are wrong.
- Check the null/NoData handling: a cell missing one criterion must not quietly
  score 0, which would read as "least suitable" instead of "unknown".
- Sanity-check the extremes: look at the highest and lowest scoring locations
  and confirm they are plausible given the inputs.

## Common failure

Combining a raster in metres with one in degrees, or layers with different
extents, so the arithmetic silently aligns the wrong cells. Print each layer's
CRS, shape and bounds before the weighted sum, and confirm they match.
