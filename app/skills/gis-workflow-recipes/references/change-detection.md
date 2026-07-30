# Recipe — same area, two dates

The shape: land cover, forest extent, built-up area or a water body at time 1
and time 2; the question is what changed, where, and by how much.

## Steps

1. **Verify the two dates are comparable.** Same CRS, same extent, and for
   rasters the same resolution and grid alignment. If they differ, reproject and
   resample the later one onto the earlier one's grid — and say you did.

2. **Restrict to the common area.** Clip both to the intersection of their
   extents. Change statistics over non-overlapping areas are meaningless.

3. **Compute change**, matching the data type:
   - categorical raster → a from–to cross-tabulation (transition matrix),
   - continuous raster → difference (t2 − t1), and where relevant a normalised
     index difference,
   - vector polygons → `difference` for loss, `intersection` for persistence,
     and the reverse `difference` for gain.

4. **Quantify**: area gained, area lost, net change, and percent of the time-1
   extent. Report all four — net change alone hides simultaneous gain and loss.

5. **Map** with a diverging scheme centred on zero for continuous change, or
   explicit gain/loss/unchanged categories for categorical change.

## Verification

- gain + unchanged + loss must reconstruct the totals. Print the arithmetic
  check; if it does not close, geometries overlap or the clip was skipped.
- For rasters, confirm both arrays have identical shape before subtracting. A
  broadcast against a mismatched shape produces a plausible-looking wrong answer.
- Confirm NoData in either date is excluded rather than treated as 0 — otherwise
  every NoData cell reads as a dramatic change.
- Compare the change magnitude against the time span. A 90% forest loss over one
  year is possible but demands a second look at the data before reporting it.

## Common failure

Comparing two rasters that look aligned but are offset by half a cell, so every
edge shows spurious change. Check the transforms are equal, not just the shapes.
