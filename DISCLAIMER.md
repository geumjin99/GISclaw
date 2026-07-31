# Disclaimer

Read this before using GISclaw for anything that matters.

## Results are produced by a language model

GISclaw plans and writes its own analysis code. The planning, the code, and the
written summary all come from a large language model, which is probabilistic: it
can be wrong, and it can be wrong while sounding entirely confident. The
operating discipline built into this software — read the schema, check the CRS,
print null rates, verify ranges before finishing — comes from roughly 1,800
controlled runs and measurably reduces these failures. **It does not eliminate
them.**

The author makes no representation that any output is correct, complete, or fit
for any purpose, and accepts no responsibility or liability for results the
model produces or for decisions taken on the basis of them.

## Check the work before you rely on it

Every run leaves the code it executed (`runs/<run>/code.py`), the full trace, and
the outputs. Read them. These are failure modes observed repeatedly in testing,
all of which produce a file that looks finished:

- A join that silently matches nothing, giving an all-null column.
- Layers combined in mismatched coordinate reference systems, so distances and
  areas are wrong by an arbitrary factor.
- An interpolated surface with values outside anything observed — the real
  example that shaped this software was a Kriging result of −27,925 °F.
- Missing values quietly filled, so a map shows a pattern that was manufactured
  rather than measured.
- A correct implementation of the wrong plan.

If a number matters, reproduce it independently.

## Not professional advice

GISclaw is a research tool. Its output is not surveying, engineering,
environmental, legal, planning, medical, or financial advice, and is not a
substitute for a qualified professional or for an authoritative data source.

Take particular care where a wrong answer causes harm — flood and hazard
mapping, emergency response, infrastructure siting, environmental impact
assessment, land tenure, public health. In those settings treat GISclaw output
as a draft for expert review, never as a finding.

## No warranty

GISclaw is distributed under the GNU Affero General Public License v3.0 or
later. As stated in sections 15 and 16 of that licence, the program comes with
**no warranty of any kind**, and no copyright holder or contributor is liable
for any damages arising from its use. Nothing here or elsewhere in this
repository grants a warranty or creates a support obligation.

## Your data leaves your machine

The analysis runs locally, but the reasoning does not. To decide what to do
next, GISclaw sends the model provider you configure a description of your
data — file names, column names, coordinate reference systems, extents, summary
statistics, error messages, and sample values that appear in program output.

If your data is confidential, restricted, or personal, satisfy yourself that
sending this material to that provider is permitted before you begin. The
provider's own terms and retention policy govern what happens to it. GISclaw
does not transmit anything anywhere else.

## Costs are yours

You supply your own API keys and are billed directly by the provider. A long or
repeatedly retried analysis costs more than a short one. Set a spending limit in
the provider's console.

## Third-party data

The bundled example dataset is included to demonstrate the software and carries
its own licence and citation requirement — see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Basemap tiles come from a
third-party service and their accuracy is not the author's to vouch for.
Anything you produce from your own data remains yours, and remains your
responsibility.
