"""GISclaw product — skills, Claude-Code-compatible.

A skill is a **directory bundle**, not a single file — the same shape the Claude
Code ecosystem uses, so a skill authored for one works here:

    <skill>/
        SKILL.md          required. YAML frontmatter + a short *router* body.
        manifest.yaml     optional. Declarative loading plan (always_load, axes…).
        references/*.md   optional. Deep material, read on demand.
        static/, assets/, scripts/, evals/   optional. Anything else the skill ships.

## Progressive disclosure

Injecting every skill body into every call does not scale — `nature-figure`
alone is a 5 KB router in front of 116 KB of references. So there are three
levels, and only the first costs tokens unconditionally:

  1. **Catalog** — `name — description` for every enabled skill, always in the
     system prompt. This is what lets the model decide a skill is relevant.
  2. **Router** — the agent calls `skill(name=...)` and gets SKILL.md plus the
     bundle's file listing.
  3. **Depth** — the agent calls `skill(name=..., path="references/api.md")` for
     one file at a time, exactly as the SKILL.md router instructs.

A skill with `always: true` in its frontmatter skips the dance: its body is
injected every run (used for standing operating rules, not procedures).

## Discovery roots

Later roots shadow earlier ones by skill name:

    app/skills/                       builtin, ships with the app — ours only
    $GISCLAW_EXTRA_SKILLS             optional, local-only escape hatch (unset by default)
    <WORKSPACE>/.gisclaw/skills/      yours: imported or authored here, editable

The app deliberately ships **no** third-party bundles and mounts no personal
skills folder: distributing someone else's skills is not ours to do. Import what
you want per install instead.
"""
import io
import os
import re
import shutil
import zipfile

SKILL_FILE = "SKILL.md"

# Text we are willing to hand to the model; anything else is listed but not read.
TEXT_EXT = {".md", ".txt", ".yaml", ".yml", ".json", ".py", ".r", ".csv", ".tsv",
            ".toml", ".cfg", ".ini", ".sh", ".sql", ".js", ".html", ".xml", ".rst"}
RESOURCE_CHAR_CAP = 24000       # ~6k tokens per read
LISTING_CAP = 200               # files shown in a bundle listing
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules"}

SKILL_TEMPLATE = """---
name: {name}
description: One line on when this applies. Write it so the agent can tell from
  this alone whether the skill is relevant — it is the only part always in context.
always: false
version: 1
---

# {title}

This body is the **router**. Keep it short: it loads only when the agent decides
the skill is relevant, and everything here competes for context with the task.

## When to use

Describe the trigger conditions in one or two lines.

## Procedure

1. First step.
2. Second step.

## Deep material

Put long content in `references/` next to this file and point at it, e.g.:

- Read `references/checklist.md` before delivering.

The agent loads those with `skill(name="{name}", path="references/checklist.md")`,
one file at a time, only when a step needs it.
"""


def _parse_frontmatter(text: str):
    """Split `---\\n…\\n---\\n` frontmatter from the body. Returns (meta, body)."""
    meta = {}
    body = text
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, flags=re.S)
    if m:
        raw, body = m.group(1), m.group(2)
        try:
            import yaml
            loaded = yaml.safe_load(raw)
            if isinstance(loaded, dict):
                meta = loaded
        except Exception:
            # A malformed header must never hide the skill — fall back to key: value.
            for line in raw.splitlines():
                if ":" in line and not line.startswith((" ", "\t", "-")):
                    k, _, v = line.partition(":")
                    meta[k.strip()] = v.strip()
    return meta, body.strip()


def _as_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "on")
    return default


def _as_list(v) -> list:
    if isinstance(v, (list, tuple)):
        return [str(x).strip().lower() for x in v if str(x).strip()]
    if isinstance(v, str):
        return [p.strip().lower() for p in re.split(r"[,;]", v) if p.strip()]
    return []


# Words too common to signal anything about which skill applies.
_STOP = {"the", "and", "for", "with", "from", "that", "this", "into", "each",
         "then", "save", "using", "use", "make", "your", "you", "please", "data",
         "file", "files", "layer", "layers", "map", "result", "results", "report",
         "compute", "calculate", "analysis", "analyse", "analyze", "would", "should",
         "what", "which", "where", "when", "all", "any", "new", "out", "per", "its"}


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z][a-z0-9_-]{3,}", (text or "").lower())
            if w not in _STOP}


def _one_line(s: str, cap: int = 400) -> str:
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s if len(s) <= cap else s[:cap - 1] + "…"


class SkillsStore:
    def __init__(self, workspace: str, builtin_dir: str, extra_dir: str = ""):
        self.user_dir = os.path.join(workspace, ".gisclaw", "skills")
        self.builtin_dir = builtin_dir
        # Local-only escape hatch: point at another skills folder on this machine.
        # Nothing sets it by default, and it must never be baked into a shipped
        # compose file — see the module docstring.
        self.extra_dir = extra_dir or os.environ.get("GISCLAW_EXTRA_SKILLS", "")
        os.makedirs(self.user_dir, exist_ok=True)

    def roots(self):
        """(source label, path) in shadowing order — later wins."""
        out = [("builtin", self.builtin_dir)]
        if self.extra_dir:
            out.append(("extra", self.extra_dir))
        out.append(("user", self.user_dir))
        return [(lbl, p) for lbl, p in out if p]

    # ------------------------------------------------------------ discovery --
    def _scan(self, root: str, source: str) -> dict:
        found = {}
        if not os.path.isdir(root):
            return found
        for entry in sorted(os.listdir(root)):
            sdir = os.path.join(root, entry)
            path = os.path.join(sdir, SKILL_FILE)
            if not os.path.isfile(path):
                continue          # a bundle without SKILL.md is not a skill
            try:
                with open(path, encoding="utf-8") as f:
                    meta, body = _parse_frontmatter(f.read())
            except Exception:
                continue
            name = str(meta.get("name") or entry).strip()
            found[name] = {
                "name": name,
                "folder": entry,
                "dir": sdir,
                "path": path,
                "description": _one_line(meta.get("description")),
                "always": _as_bool(meta.get("always")),
                "version": str(meta.get("version") or ""),
                "author": _one_line(meta.get("author"), 120),
                "license": _one_line(meta.get("license"), 60),
                "allowed_tools": meta.get("allowed-tools") or meta.get("allowed_tools") or "",
                # Optional trigger phrases. Claude-Code skills rely on the model
                # reading `description`; weaker models need something literal.
                "keywords": _as_list(meta.get("keywords")),
                "body": body,
                "source": source,
            }
        return found

    def discover(self, overrides: dict = None, with_body: bool = False) -> list:
        overrides = overrides or {}
        merged = {}
        for source, root in self.roots():
            for name, sk in self._scan(root, source).items():
                if name in merged:
                    sk["shadows"] = merged[name]["source"]
                merged[name] = sk
        out = []
        for name, sk in merged.items():
            ov = overrides.get(name, {})
            sk["enabled"] = _as_bool(ov.get("enabled"), default=sk["always"])
            sk["resources"] = self._count_resources(sk["dir"])
            sk["router_tokens_est"] = max(1, len(sk["body"]) // 4)
            # What this skill costs unconditionally: whole body if always-on,
            # otherwise just its catalog line.
            sk["always_tokens_est"] = (sk["router_tokens_est"] if sk["always"]
                                       else max(1, len(sk["description"]) // 4))
            if not with_body:
                sk.pop("body", None)
            out.append(sk)
        out.sort(key=lambda s: (not s["always"], not s["enabled"], s["name"]))
        return out

    def get(self, name: str, overrides: dict = None):
        for sk in self.discover(overrides, with_body=True):
            if sk["name"] == name:
                return sk
        return None

    # ------------------------------------------------------------- bundles --
    def _walk(self, sdir: str):
        for root, dirs, files in os.walk(sdir):
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS and not d.startswith("."))
            for fn in sorted(files):
                if fn.startswith("."):
                    continue
                full = os.path.join(root, fn)
                yield os.path.relpath(full, sdir), full

    def _count_resources(self, sdir: str) -> int:
        n = 0
        for rel, _ in self._walk(sdir):
            if rel != SKILL_FILE:
                n += 1
                if n >= LISTING_CAP:
                    break
        return n

    def list_resources(self, name: str, overrides: dict = None) -> list:
        sk = self.get(name, overrides)
        if not sk:
            return []
        items = []
        for rel, full in self._walk(sk["dir"]):
            if rel == SKILL_FILE:
                continue
            ext = os.path.splitext(rel)[1].lower()
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            items.append({"path": rel.replace(os.sep, "/"), "size": size,
                          "readable": ext in TEXT_EXT})
            if len(items) >= LISTING_CAP:
                break
        return items

    def read_resource(self, name: str, rel: str, overrides: dict = None):
        """Read one file from inside a skill bundle. Refuses to escape it."""
        sk = self.get(name, overrides)
        if not sk:
            return {"error": f"no skill named '{name}'"}
        base = os.path.abspath(sk["dir"])
        target = os.path.abspath(os.path.join(base, rel))
        if target != base and not target.startswith(base + os.sep):
            return {"error": "path escapes the skill bundle"}
        if not os.path.isfile(target):
            return {"error": f"no such file in '{name}': {rel}"}
        if os.path.splitext(target)[1].lower() not in TEXT_EXT:
            return {"error": f"'{rel}' is not a text file — it can be used by scripts, not read."}
        try:
            with open(target, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception as e:
            return {"error": f"could not read '{rel}': {e}"}
        truncated = len(text) > RESOURCE_CHAR_CAP
        if truncated:
            text = text[:RESOURCE_CHAR_CAP] + f"\n\n… [truncated at {RESOURCE_CHAR_CAP} chars]"
        return {"text": text, "truncated": truncated, "path": rel}

    # ------------------------------------------------------------ authoring --
    def write_user_skill(self, name: str, text: str) -> str:
        """Create/overwrite SKILL.md in the workspace copy of a skill."""
        existing = self._scan(self.user_dir, "user").get(name)
        folder = existing["folder"] if existing else (
            re.sub(r"[^a-zA-Z0-9_-]+", "-", name.strip()).strip("-") or "skill")
        sdir = os.path.join(self.user_dir, folder)
        os.makedirs(sdir, exist_ok=True)
        path = os.path.join(sdir, SKILL_FILE)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def fork(self, name: str) -> str:
        """Copy a whole bundle (references and all) into the workspace to edit."""
        src = None
        for source, root in self.roots():
            if source == "user":
                continue
            hit = self._scan(root, source).get(name)
            if hit:
                src = hit
        if not src:
            raise FileNotFoundError(f"no non-user skill '{name}' to fork")
        dst = os.path.join(self.user_dir, src["folder"])
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src["dir"], dst)
        return dst

    def delete_user_skill(self, name: str) -> bool:
        sk = self._scan(self.user_dir, "user").get(name)
        if not sk:
            return False
        shutil.rmtree(sk["dir"], ignore_errors=True)
        return True

    def new_skill_template(self, name: str) -> str:
        title = name.replace("-", " ").replace("_", " ").strip().title()
        return SKILL_TEMPLATE.format(name=name, title=title)

    # -------------------------------------------------------------- import --
    def _install_dir(self, src_dir: str, prefer_name: str = "") -> str:
        meta, _ = _parse_frontmatter(open(os.path.join(src_dir, SKILL_FILE),
                                          encoding="utf-8").read())
        name = str(meta.get("name") or prefer_name or os.path.basename(src_dir)).strip()
        folder = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-") or "skill"
        dst = os.path.join(self.user_dir, folder)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src_dir, dst)
        return name

    @staticmethod
    def _find_bundle_root(base: str) -> str:
        """Locate the dir holding SKILL.md — zips usually wrap it one level deep."""
        if os.path.isfile(os.path.join(base, SKILL_FILE)):
            return base
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            if SKILL_FILE in files:
                return root
        return ""

    def import_zip(self, blob: bytes) -> str:
        """Install a skill from a .zip — the ecosystem's usual transport."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            try:
                with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                    for member in zf.namelist():          # refuse zip-slip
                        p = os.path.normpath(member)
                        if p.startswith("..") or os.path.isabs(p):
                            raise ValueError(f"unsafe path in zip: {member}")
                    zf.extractall(tmp)
            except zipfile.BadZipFile:
                raise ValueError("not a valid .zip file")
            root = self._find_bundle_root(tmp)
            if not root:
                raise ValueError("no SKILL.md found anywhere in the archive")
            return self._install_dir(root)

    def import_path(self, src: str) -> str:
        """Install from a directory the server can already see (e.g. under /workspace)."""
        src = os.path.abspath(src)
        if not os.path.isdir(src):
            raise ValueError(f"not a directory: {src}")
        root = self._find_bundle_root(src)
        if not root:
            raise ValueError(f"no SKILL.md under {src}")
        return self._install_dir(root)

    def export_zip(self, name: str) -> bytes:
        """Zip a bundle back up so it can be shared or dropped into Claude Code."""
        sk = self.get(name)
        if not sk:
            raise FileNotFoundError(f"no skill '{name}'")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel, full in self._walk(sk["dir"]):
                zf.write(full, os.path.join(sk["folder"], rel))
        return buf.getvalue()

    # ------------------------------------------------------------ injection --
    def manifest_always_load(self, name: str, overrides: dict = None) -> list:
        """Files a skill's manifest.yaml declares as `always_load`.

        This is the ecosystem's own convention for "the router is not enough" —
        nature-figure, for instance, pins its contract and stance fragments this
        way. When a skill is loaded we honour it, so a bundle authored for Claude
        Code behaves the same here without the model having to fetch them.
        """
        sk = self.get(name, overrides)
        if not sk:
            return []
        mpath = os.path.join(sk["dir"], "manifest.yaml")
        if not os.path.isfile(mpath):
            return []
        try:
            import yaml
            with open(mpath, encoding="utf-8") as f:
                man = yaml.safe_load(f) or {}
        except Exception:
            return []
        out = []
        for rel in (man.get("always_load") or [])[:8]:
            res = self.read_resource(name, str(rel), overrides)
            if not res.get("error"):
                out.append({"path": str(rel), "text": res["text"]})
        return out

    def build_always_block(self, overrides: dict = None) -> str:
        """Level 0: bodies of enabled `always: true` skills — standing rules."""
        parts = []
        for sk in self.discover(overrides, with_body=True):
            if sk["enabled"] and sk["always"] and sk.get("body"):
                parts.append(f"### Skill: {sk['name']}\n\n{sk['body']}")
        return "\n\n".join(parts)

    def match(self, instruction: str, overrides: dict = None):
        """Pick the one on-demand skill this task is about, or None.

        Why this exists: the ecosystem assumes the model reads `description` and
        invokes the skill itself. Measured here, DeepSeek-class models do not —
        they start analysing immediately and never call the tool, which is this
        project's own finding that weaker models can code but will not plan the
        route. So the router is chosen server-side and pre-loaded for them; the
        model can still go deeper on its own with `skill(name, path)`.

        Declared `keywords` win outright; otherwise it takes real vocabulary
        overlap with name + description, so an unrelated task matches nothing.
        """
        text = (instruction or "").lower()
        itoks = _tokens(text)
        best, best_score = None, 0
        for sk in self.discover(overrides, with_body=True):
            if not sk["enabled"] or sk["always"]:
                continue
            score = 0
            for kw in sk.get("keywords", []):
                if (" " in kw and kw in text) or (kw in itoks):
                    score += 5
            score += 2 * len(_tokens(sk["name"]) & itoks)
            score += len(_tokens(sk["description"]) & itoks)
            if score > best_score:
                best, best_score = sk, score
        if best and best_score >= 5:
            return dict(best, match_score=best_score)
        return None

    def build_catalog(self, overrides: dict = None, exclude: str = "") -> str:
        """Level 1: one line per on-demand skill — the only thing always paid for."""
        rows = []
        for sk in self.discover(overrides):
            if not sk["enabled"] or sk["always"] or sk["name"] == exclude:
                continue
            extra = f" [{sk['resources']} bundled file(s)]" if sk["resources"] else ""
            rows.append(f"- **{sk['name']}** — {sk['description'] or 'no description'}{extra}")
        return "\n".join(rows)
