---
name: gis-workflow-recipes
description: Step-by-step workflow templates for recurring multi-step GIS analyses — point interpolation with zonal summary, multi-criteria suitability overlay, proximity/impact assessment, and two-date change detection. Use when the task matches one of these shapes and you would otherwise have to invent the analysis path yourself.
always: false
version: 1
author: GISclaw, distilled from the GeoAnalystBench experiments
keywords:
  - interpolate
  - interpolation
  - kriging
  - zonal
  - choropleth
  - suitability
  - weighted overlay
  - multi-criteria
  - risk score
  - buffer
  - proximity
  - impact
  - affected area
  - change detection
  - land cover change
  - two dates
---

# GIS workflow recipes — router

The experiments behind this system found that the hard part of a multi-step GIS
task is **not writing the code — it is knowing which steps to write**. Weaker
models collapse without a template and work well with one; stronger models get a
smaller but real benefit. This skill is that template library.

Do not work from this router alone. It only tells you which recipe to load.

## Choosing a recipe

Match the task to one shape, then read exactly one file:

| The task looks like | Read |
|---|---|
| Scattered measurements → continuous surface → summarise per polygon | `references/interpolation-zonal.md` |
| Combine several weighted criteria into a suitability/risk score | `references/suitability-overlay.md` |
| What does a line/point feature affect within some distance | `references/proximity-impact.md` |
| Compare the same area at two dates | `references/change-detection.md` |

If none matches, do not force one — plan the task yourself and follow the
standing operating discipline instead.

## How to use a recipe

1. Read the one matching file with
   `skill(name="gis-workflow-recipes", path="references/<file>.md")`.
2. Restate the recipe's steps as a concrete plan **against the actual layers and
   field names in this project** — never against the recipe's placeholder names.
   Inspect the data first if you have not already.
3. Drop steps the task does not need, and say in your Thought which you dropped
   and why. A recipe is a checklist, not a script to execute blindly.
4. Every recipe ends with a verification block. That part is not optional.

## What recipes deliberately do not do

They do not choose your CRS, class breaks, or weights for you — those depend on
the data and the client. Where a recipe says "choose", make the choice
explicitly and record the reason in your Thought so it lands in the run record.
