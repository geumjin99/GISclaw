---
name: gis-analysis-discipline
description: Core operating discipline for multi-step GIS analysis — plan, inspect, CRS, verify. Distilled from ~1800 controlled agent runs on GeoAnalystBench.
always: true
version: 1
---

# Operating discipline

These rules come from ~1800 controlled runs of this agent on real multi-step GIS
tasks. Each one exists because ignoring it produced a wrong answer that *looked*
right. Follow them even when the task seems simple.

## 1. Decide the whole path before writing any code

The dominant failure mode is not bad Python — it is **correct code for the wrong
plan**. In your first Thought, write the full chain: inputs → operations → the
exact artefacts you must produce. If a step's output does not feed a later step
or a deliverable, delete it.

If the task is ambiguous, choose the interpretation that produces a checkable
number, and state that choice in the first Thought.

## 2. Look before you compute

Never assume a schema. Before the first real operation:

- `list_files`, then load every input.
- Print, for each layer: feature count, geometry type, CRS, and the **actual
  column names** (`list(gdf.columns)`).
- Print the range of any field you are about to use (`min`, `max`, null count).

Hardcoded column names that "should" exist are the single most common cause of a
run dying at round 3. Read them from the data.

## 3. CRS is the most expensive mistake

- Input layers routinely have **different** CRS. Check every one; never assume
  they match.
- Any distance, area, buffer, or nearest-neighbour work requires a **projected**
  CRS in metres — not EPSG:4326. Reproject first, and say which CRS you chose and
  why in the Thought.
- When layers disagree, reproject to the CRS of the layer that defines the
  analysis units (usually the polygons you report on), not to whatever is first.
- After a spatial join, print the null rate of the joined fields. A silent 100%
  null join is the classic "it ran fine" disaster — it means the CRS or the
  predicate is wrong.

## 4. Prefer the deterministic operations

For anything the built-in `geoprocess` operations cover — reproject, buffer,
clip, intersection, difference, union, dissolve, spatial/attribute join,
zonal statistics, slope/aspect/hillshade, rasterize, IDW — use them. They are
CRS-aware, tested, and reproducible. Hand-written equivalents in `execute()` are
where subtle errors enter. Use `execute()` for what no operation covers: custom
formulas, modelling, and plots.

## 5. Verify the numbers before you finish

"Finished with files in `pred_results/`" is **not** success. Before `finish`,
print and sanity-check:

- **Counts**: is the output feature count plausible? Zero rows, or exactly as
  many rows as the input when a filter was applied, means the logic is wrong.
- **Ranges**: do the values sit in a physically possible band? Interpolation is
  the worst offender — an unconstrained Kriging surface once returned −27925 °F.
  Compare the output range against the input range; a surface should not wander
  far outside the observed data.
- **Rasters**: check it is not all-zero, all-NaN, or a single constant.
- **Figures**: confirm the plotted field actually varies. A uniformly coloured
  choropleth almost always means the join produced nulls, not that the world is
  uniform.

State the checked numbers in the Thought. If a check fails, fix it — do not
finish and mention the problem in passing.

## 5a. Never invent values for missing data

Units with no measurement stay **null**. Do not fill them with the mean, with
zero, or with a neighbour's value to make a join, a score, or a map look
complete — a filled value is indistinguishable from a measured one downstream,
and it silently becomes evidence.

If a computation genuinely cannot proceed without filling:

1. Say so in the Thought **before** doing it, with the reason.
2. Report how many units were filled, out of how many, and with what.
3. Keep a flag column marking which rows were imputed.
4. Say it again in the final summary — not only in a print statement.

A result covering 99 of 269 units and saying so is worth more than one covering
269 units where 170 are fabricated.

## 6. When something errors, diagnose rather than retry

The real error is at the **end** of a traceback. Read it, form a hypothesis,
and change one thing. Re-running the same code, or rewriting the whole block
from scratch, wastes rounds and usually reintroduces the bug. Never silently
swallow an exception with a bare `try/except` to make a step "pass".

## 7. Protect state and outputs

- The sandbox namespace persists between calls. **Never overwrite an original
  loaded variable** — derive new names (`blocks_utm`, not `blocks`). Later steps
  may need the original.
- Write every deliverable into `pred_results/`, then `print(os.listdir('pred_results/'))`
  to prove it landed.
- Save figures with `plt.savefig(...)` inside the **same** `execute()` call that
  creates them. Never call `plt.show()`.

## 8. Report only what you measured

When summarising, cite numbers you actually printed in this run. Do not
approximate, round away precision, or restate the task's assumptions as
findings. If something could not be computed, say so plainly.
