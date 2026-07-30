# Recipe — what does this feature affect within some distance

The shape: a planned road, pipeline, factory or well, and a question about what
it disturbs — protected forest, housing, habitat, a watershed.

## Steps

1. **Inspect.** Geometry type of the impacting feature (line? point? polygon?),
   geometry of the affected layer, CRS of both, and any attribute that defines
   protection class or sensitivity.

2. **Project to a metric CRS.** Buffers in EPSG:4326 are in *degrees* and are
   silently wrong — a 1000 "unit" buffer becomes ~111 km. Use the `buffer` op,
   which estimates a suitable UTM zone when the input is geographic, or
   reproject explicitly first and say which CRS you chose.

3. **Buffer at the distances the task names.** If it names several (e.g. 100 m,
   500 m, 1 km), build them all; do not collapse to one. If it names none,
   choose one, state it, and justify it in a sentence.

4. **Intersect** the buffer with the affected layer. Use the `intersection` op,
   which keeps attributes from both sides — you need them for step 5.

5. **Quantify the impact.** Not just "yes it overlaps":
   - affected **area** (compute in the projected CRS, report in km² or ha),
   - **share** of each affected polygon that falls inside the buffer,
   - **counts** per protection class or category,
   - the share of the total protected area affected.

6. **Map it**: the feature, its buffer(s), and the affected polygons highlighted,
   with the untouched layer visible underneath for context.

## Verification

- Area units: confirm you computed area **after** projecting, and report the
  unit explicitly. An area of 1e-5 means you computed in degrees.
- The affected area must be ≤ the total area of the affected layer, and ≤ the
  buffer's own area. If it is not, geometries are invalid or CRS differ — run a
  `buffer(0)`-style validity fix, or check `.is_valid`.
- Zero overlap is a real possible answer, but verify it: print whether the two
  layers' bounds intersect at all before reporting "no impact".

## Common failure

Reporting overlap counts without area. "12 forest polygons affected" is not an
impact assessment — 12 polygons could be 0.4 ha or 400 km². Always give area.
