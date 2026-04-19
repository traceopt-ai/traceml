# Developer Docs Site — Design Spec

**Date:** 2026-04-19
**Branch:** `docs/dev-docs-site`
**Author:** Abhijeet Pendyala
**Status:** Approved — ready for implementation plan

---

## Context

TraceML (`traceml-ai` on PyPI, v0.2.9) has no developer documentation site. Existing user-facing markdown lives as loose files in `docs/`: `quickstart.md`, `faq.md`, `huggingface.md`, `lightning.md`, `how-to-read-output.md`, `use-with-wandb-mlflow.md`, plus an architecture PNG. GitHub renders these, but there is no unified site, no auto-generated API reference, no navigation, no search, no consistent theme.

The goal is an MVP developer docs site to (a) demonstrate the project to stakeholders (Hima, investors) and (b) give contributors a browsable entry point into the codebase.

The original Ecoki project used Sphinx + reStructuredText + Google docstrings + LaTeX-branded PDFs. This project deliberately picks a different stack (MkDocs Material + mkdocstrings + Markdown) because:

- The codebase and existing docs are Markdown-native.
- MkDocs Material is the dominant choice among modern Python libraries (FastAPI, Pydantic, HTTPX).
- No LaTeX build pipeline, no RST learning curve for contributors.

A portable blueprint of the Ecoki workflow exists at `DEV_DOCS_BLUEPRINT.md` for reference — the outcomes transfer, the tooling does not.

## Goals

- Deploy a public docs site to GitHub Pages on every push to `main`.
- Unified site with two top-level sections: **User Guide** (existing content) and **Developer Guide** (architecture + internals).
- Auto-generated API reference for both public API (what users call) and curated developer internals (the 11 architecture layers).
- Markdown-native authoring, NumPy-style docstring parsing, Mermaid diagram support, live-reload dev loop.
- Cross-page API references resolvable via `autorefs` (internal "intersphinx" equivalent).
- CI fails PRs that break links or rendering (`mkdocs build --strict`), with pip caching for fast feedback and concurrency-cancelled PR builds on rebase.
- Pre-push hook enforces `mkdocs build --strict` locally so contributors catch breaks before pushing.
- Issue templates (bug / feature / design doc), PR template with docs checklist, and CODEOWNERS routing all ported from the Ecoki blueprint and adapted for GitHub.

## Non-Goals (deferred / backlog)

- PDF output — HTML only. Browser "Save as PDF" is sufficient.
- Integrating `Notes/Code_base/` (CODE_MAP, CODE_WALKTHROUGH, METRICS_REFERENCE, INTEGRATION_REFERENCE, EXAMPLES_WALKTHROUGH, ARCHITECTURE_DIAGRAMS + SVGs) — post-MVP polish.
- Internal-only private site (`mkdocs.internal.yml` + `docs_internal/` for TIMELINE and strategic notes) — deferred.
- Custom domain (`docs.traceopt.ai` via CNAME) — deferred.
- Versioned docs via `mike` — no tagged releases worth preserving yet.
- Spell-check / vale linters in CI.
- Full docstring coverage (currently ~65% functions, ~58% modules) — must be discussed with Abhinav as a separate roadmap item.
- Migration from NumPy-style to Google-style docstrings — no value, mkdocstrings supports NumPy natively.

## Tooling stack

Dependencies added as a new `docs` extra in `pyproject.toml`:

```toml
[project.optional-dependencies]
docs = [
    "mkdocs>=1.6",
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.26",
    "mkdocs-autorefs>=1.2",
    "mkdocs-include-markdown-plugin>=7.0",
    "pymdown-extensions>=10.0",
]
```

- `mkdocs` — site generator (native CLI handles build + serve).
- `mkdocs-material` — theme; provides search, dark mode, code tabs, admonitions.
- `mkdocstrings[python]` — auto-API from docstrings, configured for NumPy style.
- `mkdocs-autorefs` — cross-page reference resolution. Lets prose link `[trace_step][]` or `[Database][traceml.database.Database]` and resolve to the right API page. Equivalent of Sphinx's intersphinx for internal refs.
- `mkdocs-include-markdown-plugin` — reuse README snippets inside docs without duplication.
- `pymdown-extensions` — Material's standard markdown extensions (tabs, admonitions, content tabs).

**Docs-build requires `torch` installed.** TraceML modules (`decorators.py`, `integrations/*.py`, layer hooks, many samplers) `import torch` at module top. mkdocstrings imports every module it documents, so building docs without torch installed fails at import. CI installs `.[docs,torch,lightning,hf]` to cover every importable module. This is the MkDocs-equivalent of Sphinx's `mock_heavy_deps.py` problem — it cannot be ignored.

Explicitly **not** adopted from the Ecoki blueprint:
- No custom `build_docs.py` — `mkdocs build` / `mkdocs serve` cover the use cases.
- No custom watcher — `mkdocs serve` auto-reloads natively.
- No RST underline fixer — project is Markdown.
- No `mock_heavy_deps.py` — instead, CI installs the optional deps (see above). Simpler and produces real API output.
- No LaTeX `.sty` files — PDF is out of scope.

## Architecture — site structure

Single `mkdocs.yml` at repo root. Site content at `docs/`.

```
docs/
├── index.md                        # Landing: pitch, install, "I want to..." links
├── user_guide/
│   ├── quickstart.md               # Moved from docs/quickstart.md
│   ├── reading-output.md           # Moved from docs/how-to-read-output.md
│   ├── integrations/
│   │   ├── huggingface.md          # Moved from docs/huggingface.md
│   │   ├── lightning.md            # Moved from docs/lightning.md
│   │   └── wandb-mlflow.md         # Moved from docs/use-with-wandb-mlflow.md
│   ├── faq.md                      # Moved from docs/faq.md
│   └── public-api.md               # NEW — mkdocstrings blocks for user-facing API
└── developer_guide/
    ├── architecture.md             # NEW — 10-layer overview + Mermaid data-flow diagram
    ├── subsystems/
    │   ├── cli.md                  # NEW — intro + ::: traceml.cli
    │   ├── runtime.md              # NEW — intro + ::: traceml.runtime
    │   ├── aggregator.md           # NEW — intro + ::: traceml.aggregator
    │   ├── samplers.md             # NEW — intro + ::: traceml.samplers
    │   ├── database.md             # NEW — intro + ::: traceml.database
    │   ├── transport.md            # NEW — intro + ::: traceml.transport
    │   ├── renderers.md            # NEW — intro + ::: traceml.renderers
    │   ├── display-drivers.md      # NEW — intro + ::: traceml.aggregator.display_drivers
    │   ├── decorators.md           # NEW — intro + ::: traceml.decorators
    │   ├── integrations.md         # NEW — intro + ::: traceml.integrations
    │   └── utils.md                # NEW — intro + ::: traceml.utils
    └── contributing.md             # NEW — commit/docstring/line-length rules
```

Internal links inside the moved files are updated to new relative paths. No content rewrites on day one — pure relocation.

**Public API page** (`docs/user_guide/public-api.md`) covers:
- `traceml.trace_step` (decorator)
- `traceml.trace_model_instance` (decorator)
- `traceml.TraceMLTrainer` (HF integration)
- `traceml.TraceMLCallback` (Lightning integration)
- CLI subcommands (`traceml watch|run|deep`)

**Developer subsystem pages** are thin by design: a 2-3 sentence intro explaining the layer's responsibility, followed by one `::: traceml.<module>` block. mkdocstrings handles the rest. Layers mirror the architecture breakdown in the root `CLAUDE.md`.

**Architecture overview page** contains:
- One Mermaid diagram of the three-process model (CLI spawns aggregator + training).
- One Mermaid diagram of the telemetry data flow (sampler → database → TCP → aggregator → renderer).
- One-paragraph description per layer.

## `mkdocs.yml` outline

```yaml
site_name: TraceML
site_description: Real-time bottleneck finder for PyTorch training runs
site_url: https://traceopt-ai.github.io/traceml/
repo_url: https://github.com/traceopt-ai/traceml
repo_name: traceopt-ai/traceml
edit_uri: edit/main/docs/

theme:
  name: material
  # logo: assets/logo.png        # add when traceopt.ai logo is available
  # favicon: assets/favicon.png  # 32x32 ico/png
  palette:
    - scheme: default
      primary: indigo
      toggle: { icon: material/brightness-7, name: Dark mode }
    - scheme: slate
      primary: indigo
      toggle: { icon: material/brightness-4, name: Light mode }
  features:
    - navigation.instant           # SPA-feel prefetching
    - navigation.instant.progress  # loading bar
    - navigation.tracking          # URL updates on scroll
    - navigation.tabs
    - navigation.sections
    - navigation.top
    - search.suggest
    - search.highlight
    - content.code.copy
    - content.code.annotate
    - content.tabs.link
    - content.action.edit          # "edit this page" pencil

plugins:
  - search
  - autorefs
  - include-markdown
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: numpy
            show_source: true
            show_signature_annotations: true
            separate_signature: true
            members_order: source
            merge_init_into_class: true
            show_root_heading: true
            show_symbol_type_heading: true

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed: { alternate_style: true }
  - pymdownx.highlight
  - pymdownx.inlinehilite
  - toc: { permalink: true }

nav:
  - Home: index.md
  - User Guide:
      - Quickstart: user_guide/quickstart.md
      - Reading output: user_guide/reading-output.md
      - Integrations:
          - Hugging Face: user_guide/integrations/huggingface.md
          - PyTorch Lightning: user_guide/integrations/lightning.md
          - W&B / MLflow: user_guide/integrations/wandb-mlflow.md
      - FAQ: user_guide/faq.md
      - Public API: user_guide/public-api.md
  - Developer Guide:
      - Architecture: developer_guide/architecture.md
      - Subsystems:
          - CLI: developer_guide/subsystems/cli.md
          - Runtime: developer_guide/subsystems/runtime.md
          - Aggregator: developer_guide/subsystems/aggregator.md
          - Samplers: developer_guide/subsystems/samplers.md
          - Database: developer_guide/subsystems/database.md
          - Transport: developer_guide/subsystems/transport.md
          - Renderers: developer_guide/subsystems/renderers.md
          - Display drivers: developer_guide/subsystems/display-drivers.md
          - Decorators: developer_guide/subsystems/decorators.md
          - Integrations: developer_guide/subsystems/integrations.md
          - Utils: developer_guide/subsystems/utils.md
      - Contributing: developer_guide/contributing.md
```

The spec lives at `specs/` at the repo root, outside `docs/`, so it is not picked up by `mkdocs build` and does not appear in the site.

## CI / deployment

New workflow: `.github/workflows/docs.yml`.

```yaml
name: Docs
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: docs-${{ github.ref }}
  cancel-in-progress: ${{ github.event_name == 'pull_request' }}

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0             # full history for any future git-log plugins
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
          cache: pip
          cache-dependency-path: pyproject.toml
      - run: pip install -e ".[docs,torch,lightning,hf]"
      - run: mkdocs build --strict
      - uses: actions/upload-pages-artifact@v3
        with:
          path: site
  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- PR builds run `--strict` — broken internal links, missing references, or malformed markdown fail the check.
- Only pushes to `main` deploy. PRs build but do not deploy.
- `pip` cache keyed on `pyproject.toml` hash — after first CI run, subsequent runs reuse wheels (build time ~60s → ~15s).
- `concurrency` cancels superseded PR builds on rebase; main-branch deploys are never cancelled.
- CI installs `[docs,torch,lightning,hf]` extras (not just `[docs]`) — required so mkdocstrings can import every TraceML module without `ImportError`.
- GitHub Pages must be enabled in repo settings (source: "GitHub Actions") as a one-time setup step.

## Issue templates

Ported from the Ecoki blueprint into `.github/ISSUE_TEMPLATE/`:

- `bug_report.md` — feature tested, description, repro steps, expected, actual, system, environment, owner, associated feature.
- `feature_request.md` — concept, inputs, outputs, business criteria, non-functional criteria, quality criteria, owner, associated requirements.
- `design_doc.md` — objective, scope, components/architecture, test cases/implementation plan, tools and libraries, execution plan, conclusion.

YAML frontmatter (`name`, `about`, `labels`, `title`) added so GitHub's new-issue chooser displays them correctly.

## Repo-root additions

**`.gitignore`** — append:
```
site/
.cache/
```

**`.github/pull_request_template.md`** — every PR gets an auto-populated checklist:
```markdown
## Summary
<!-- what changed, why -->

## Checklist
- [ ] Tests added or updated (unless docs-only)
- [ ] Docs updated if user-facing behavior changed
- [ ] `mkdocs build --strict` succeeds locally
- [ ] Commit messages are short single lines, no Co-Authored-By trailers
- [ ] New public functions/classes have NumPy docstrings (Parameters / Returns / Raises)
- [ ] Internal doc links resolve (`mkdocs serve` renders clean)
```

**`.github/CODEOWNERS`** — auto-requests review from the right person:
```
# Default owner
*                           @abhinav
# Docs owned by Abhijeet
/docs/                      @Pendu
/mkdocs.yml                 @Pendu
/.github/workflows/docs.yml @Pendu
```
(Adjust GitHub handles as needed.)

**`.pre-commit-config.yaml`** — append a docs hook so `mkdocs build --strict` runs locally before push:
```yaml
- repo: local
  hooks:
    - id: mkdocs-build
      name: mkdocs build --strict
      entry: mkdocs build --strict --quiet
      language: system
      pass_filenames: false
      files: ^(docs/|mkdocs\.yml|src/)
      stages: [pre-push]
```
`pre-push` stage (not `pre-commit`) so every commit isn't slowed down by a full docs build.

## Contributing page content

Short page. The Ecoki full style guide is overkill for an open-source PyTorch library. Cover:

1. **Dev setup** — `pip install -e ".[dev,torch,lightning,hf,docs]"` (all extras so the docs build works end-to-end).
2. **Branch naming** — `feature/<short-name>`, `fix/<short-name>`, `docs/<short-name>` (matches existing branch patterns).
3. **Commit messages** — short single-line, no `Co-Authored-By` trailers (per root `CLAUDE.md`).
4. **Code style** — black (line length 79), ruff, isort (black profile), pre-commit hooks.
5. **Docstrings** — NumPy style. Every public class, function, method gets one with Parameters / Returns / Raises sections. Template:

   ```python
   def trace_step(model: nn.Module):
       """Mark a training step boundary.

       Responsibilities
       ----------------
       - Marks the semantic start/end of a training step.
       - Attributes step-scoped timing events.
       - Advances the global step counter.

       Parameters
       ----------
       model : torch.nn.Module
           The model being trained. Used for memory-tracker attachment.

       Yields
       ------
       None
           Context-manager protocol; no value yielded.

       Raises
       ------
       RuntimeError
           If called outside of a training loop context.
       """
   ```
6. **Tests** — `pytest tests/`. New code must include tests unless docs-only.
7. **Docs** — if a code change affects user-facing behavior, update the relevant doc in the same PR.
8. **Local preview** — `mkdocs serve` at repo root, open http://127.0.0.1:8000.
9. **Before pushing** — `mkdocs build --strict` must succeed (pre-push hook enforces this).

## Gotchas carried over from Ecoki

Relevant ones only:

- **Commit docs with code, always** — #1 source of doc debt is "I'll update the docs later". Build the expectation into the review checklist.
- **`site/` out of git** — MkDocs' generated output directory. Add to `.gitignore`.
- **Relative doc paths only** — survive moves.
- **Compress images before commit** — the existing `traceml_architecture_diagram.png` is 730KB; leave it alone for MVP but flag as backlog item.
- **Live reload is a force multiplier** — document `mkdocs serve` prominently on the contributing page.

Sphinx-specific gotchas (LaTeX passes, WebP images, underline drift, `mock_heavy_deps`) do not apply.

## Out of scope — explicit backlog

Captured so the roadmap can pick these up intentionally. Grouped by type.

### Content & structure

1. **Full docstring coverage.** Today ~65% of functions documented; private helpers and some modules have none. Requires roadmap buy-in from Abhinav. Likely 1-2 phases of dedicated work per subsystem.
2. **Notes/Code_base/ integration.** CODE_MAP, CODE_WALKTHROUGH, INTEGRATION_REFERENCE, METRICS_REFERENCE, EXAMPLES_WALKTHROUGH, ARCHITECTURE_DIAGRAMS — ~140KB of existing high-quality content. Integrating them transforms reference density of both User Guide and Developer Guide. Post-MVP phase.
3. **Internal-only MkDocs site.** `mkdocs.internal.yml` + `docs_internal/` for TIMELINE.md and strategic notes. Same Material theme, never deployed, served locally via `mkdocs serve -f mkdocs.internal.yml`. ~30 min once content is ready.
4. **Architecture Decision Records.** `docs/adr/` for permanent rationale — distinct from `specs/` (ephemeral implementation planning).
5. **Glossary page** for ML/training terms (tensor parallelism, FSDP, gradient accumulation) — helps non-ML-specialist readers.
6. **Changelog integration.** Render `CHANGELOG.md` as a docs page; optionally auto-generate via `release-please`.
7. **Retire 730KB `traceml_architecture_diagram.png`** — replace with Mermaid once content is migrated.

### CI/CD hardening

8. **PR preview deploys.** GitHub Pages doesn't do this natively. Options: Cloudflare Pages (free, per-PR subdomain), Netlify (free tier), or self-hosted surge. Critical once external PRs arrive.
9. **Branch protection on `main`** — require `Docs` check to be green before merge.
10. **Weekly scheduled rebuild** — cron-triggered workflow catches rot in external references (PyPI badges, PyTorch intersphinx, changed dependencies).
11. **Dependabot for docs deps.** Add `docs` dependency group to `.github/dependabot.yml`.
12. **Broken external link checker** — `linkchecker` or `lychee` weekly.
13. **Release automation.** `release-please` or `python-semantic-release` — auto-version bumps, tag, publish to PyPI, update docs.

### Distribution & access

14. **Custom domain `docs.traceopt.ai`** — CNAME file + DNS.
15. **Versioned docs** via `mike`. Needed once tagged releases carry API differences worth preserving.
16. **Redirects plugin** (`mkdocs-redirects`) — preserve old URLs after content moves; essential post-public-launch.
17. **Analytics** — Plausible (privacy-first) preferred over Google Analytics. `extra.analytics` block.
18. **robots.txt / sitemap** — Material auto-generates sitemap; add robots.txt for crawler control.
19. **PDF output** if a stakeholder demands it later.

### Polish & SEO

20. **Social cards** — Material `social` plugin auto-generates branded OpenGraph images per page. Every LinkedIn/Twitter share gets a TraceML card.
21. **Announcement bar** — Material's `extra.announcement` block for "TraceML v1.0 is out" style notices.
22. **Custom 404 page** branded.
23. **Logo + favicon** — commit `docs/assets/logo.png` + `docs/assets/favicon.png`, uncomment the `theme.logo` / `theme.favicon` entries in `mkdocs.yml`. Blocked on having a logomark from traceopt.ai.
24. **"Last updated" timestamps** via `mkdocs-git-revision-date-localized-plugin`. Already enabled by `fetch-depth: 0` in CI.
25. **Blog plugin** (Material native) — once TraceML starts shipping posts.
26. **Footer social links** — GitHub, Discord, LinkedIn, X. `extra.social` block.
27. **Privacy / offline plugin** — bundles external assets for GDPR, EU users.

### Quality gates

28. **Vale prose linter** — modern superset of codespell + spelling. Catches passive voice, jargon, inconsistent terminology.
29. **`interrogate`** — measure docstring coverage in CI, fail below a ratcheting floor. Pairs with item #1.
30. **Markdown linter** — `markdownlint-cli2` pre-commit for consistent heading levels, list style, trailing whitespace.
31. **Image optimization** — `optipng` / `mozjpeg` pre-commit to prevent bloat.
32. **Spell-check** — `codespell` on docs content in CI.

## Acceptance criteria

1. `pip install -e ".[docs,torch,lightning,hf]"` installs all docs dependencies cleanly.
2. `mkdocs serve` launches a site at `http://127.0.0.1:8000` with the nav tree above, no build warnings.
3. `mkdocs build --strict` exits 0 on a clean checkout.
4. User Guide pages render with all internal links resolved.
5. Public API page renders auto-generated signatures for `trace_step`, `trace_model_instance`, `TraceMLTrainer`, `TraceMLCallback`, CLI entry point.
6. Every Developer Guide subsystem page renders at least one auto-generated class or function from its module.
7. Architecture page renders the Mermaid diagram visually (not as raw code).
8. `.github/workflows/docs.yml` builds green on this branch's PR.
9. `.github/ISSUE_TEMPLATE/` templates appear in GitHub's new-issue picker after merge.
10. `.github/pull_request_template.md` populates the PR body on new PRs.
11. `.github/CODEOWNERS` routes review requests correctly on a test PR touching both `src/` and `docs/`.
12. Pre-push hook `mkdocs build --strict` blocks a push that breaks the docs build.
13. CI pip cache hit observed on second run (build time drops measurably).
14. The PR description lists the backlog items above so Abhinav sees them at review time.

## Out-of-scope verification note

Because the existing upstream `origin/main` has diverged from `upstream/main` (observed a `ci.yml` merge conflict during initial rebase), this branch is based on `upstream/main`, not `origin/main`. Push target is the user's fork (`origin`) with `git push -u origin docs/dev-docs-site`. The eventual PR is against `upstream/main`.
