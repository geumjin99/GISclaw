# Contributing to GISclaw

Contributions are welcome — bug reports, fixes, new GIS operators, new skills,
translations, documentation.

If you plan to open a pull request, the licensing section below explains what
you would be agreeing to, and why it is worded the way it is.

---

## Licensing of contributions

GISclaw is licensed to the public under the **AGPL-3.0-or-later**, and is also
offered under a separate commercial licence (see
[`COMMERCIAL-LICENSE.md`](COMMERCIAL-LICENSE.md)). That second option is only
possible while a single party holds the rights to the whole work.

So, by submitting a contribution you agree to the following.

### Developer Certificate of Origin

Every commit must carry a `Signed-off-by` line:

```
git commit -s -m "your message"
```

This certifies that you wrote the contribution, or have the right to submit it,
as described by the [Developer Certificate of Origin 1.1](https://developercertificate.org/).

### Licence grant

> You retain copyright in your contribution. You grant Han Jinzhen a
> perpetual, worldwide, non-exclusive, royalty-free, irrevocable licence to
> use, reproduce, modify, prepare derivative works of, publicly display,
> sublicense, and distribute your contribution and derivative works thereof,
> **under the AGPL-3.0-or-later and under any other licence terms, including
> proprietary terms**.
>
> You confirm that you are legally entitled to grant this — in particular, that
> if your employer or institution has rights in the work, you have their
> permission.

State your agreement in the pull request with:

```
I have read CONTRIBUTING.md and I agree to the licence grant for this contribution.
```

**What this means in practice.** You keep your copyright — this is a licence,
not a transfer. Your contribution is published to everyone under the AGPL, the
same terms as the rest of the project. The grant additionally lets the project
include your contribution in commercially licensed copies, which is what funds
continued development. It gives no rights over anything else you write.

If you would rather not grant this, that is completely fine — please still open
an issue describing the problem or the idea. Bug reports need no agreement at
all.

---

## Practical guidelines

**Keep the research chain intact.** `src/agent/` is shared with the code behind
the GISclaw paper. Changes there must be backwards compatible: the ReAct agent's
6-tool contract, `on_step` defaulting to `None`, and the observation-truncation
behaviour (stdout truncated from the front, stderr from the tail) are all
load-bearing. Product-specific behaviour belongs in `app/`.

**New GIS operators** go in `src/agent/geo_ops.py` with an entry in `SPECS` so
they appear in the Toolbox UI automatically. Prefer deterministic operators over
asking the LLM to write the code.

**New skills** are directory bundles under `app/skills/<name>/` — `SKILL.md` as
the router plus optional `references/`. Keep the router small; put detail in
references so it loads only when needed.

**Match the surrounding style.** Comments in `src/agent/` are mixed
Chinese/English; follow whatever the file you are editing already does.

**Test before submitting.** There is no automated suite. Run the app
(`docker compose up`), create a project, load the bundled Madison example, and
confirm a real run completes end to end.

**Do not commit secrets.** API keys belong in `.env` or the in-app Settings
panel, never in the repository. Check your diff before pushing.

## Reporting a security issue

Please do not open a public issue. Email <hanjinzhen9@gmail.com> with details.
