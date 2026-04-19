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
- CI fails PRs that break links or rendering (`mkdocs build --strict`).
- Issue templates (bug / feature / design doc) ported from the Ecoki blueprint.

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
    "mkdocs-include-markdown-plugin>=7.0",
    "pymdown-extensions>=10.0",
]
```

- `mkdocs` — site generator (native CLI handles build + serve).
- `mkdocs-material` — theme; provides search, dark mode, code tabs, admonitions.
- `mkdocstrings[python]` — auto-API from docstrings, configured for NumPy style.
- `mkdocs-include-markdown-plugin` — reuse README snippets inside docs without duplication.
- `pymdown-extensions` — Material's standard markdown extensions (tabs, admonitions, content tabs).

Explicitly **not** adopted from the Ecoki blueprint:
- No custom `build_docs.py` — `mkdocs build` / `mkdocs serve` cover the use cases.
- No custom watcher — `mkdocs serve` auto-reloads natively.
- No RST underline fixer — project is Markdown.
- No `mock_heavy_deps.py` — mkdocstrings imports modules lazily and the Material theme isn't sensitive to side-effect-heavy imports the way Sphinx autodoc is; `pip install -e ".[docs,torch]"` in CI is straightforward.
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
  palette:
    - scheme: default
      primary: indigo
      toggle: { icon: material/brightness-7, name: Dark mode }
    - scheme: slate
      primary: indigo
      toggle: { icon: material/brightness-4, name: Light mode }
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.top
    - search.suggest
    - content.code.copy
    - content.code.annotate
    - content.tabs.link

plugins:
  - search
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
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install -e ".[docs]"
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
- GitHub Pages must be enabled in repo settings (source: "GitHub Actions") as a one-time setup step.

## Issue templates

Ported from the Ecoki blueprint into `.github/ISSUE_TEMPLATE/`:

- `bug_report.md` — feature tested, description, repro steps, expected, actual, system, environment, owner, associated feature.
- `feature_request.md` — concept, inputs, outputs, business criteria, non-functional criteria, quality criteria, owner, associated requirements.
- `design_doc.md` — objective, scope, components/architecture, test cases/implementation plan, tools and libraries, execution plan, conclusion.

YAML frontmatter (`name`, `about`, `labels`, `title`) added so GitHub's new-issue chooser displays them correctly.

## Contributing page content

Short page. The Ecoki full style guide is overkill for an open-source PyTorch library. Cover:

1. **Dev setup** — `pip install -e ".[dev,torch,docs]"`.
2. **Branch naming** — `feature/<short-name>`, `fix/<short-name>`, `docs/<short-name>` (matches existing branch patterns).
3. **Commit messages** — short single-line, no `Co-Authored-By` trailers (per root `CLAUDE.md`).
4. **Code style** — black (line length 79), ruff, isort (black profile), pre-commit hooks.
5. **Docstrings** — NumPy style. Every public class, function, method gets one with Parameters / Returns / Raises sections.
6. **Tests** — `pytest tests/`. New code must include tests unless docs-only.
7. **Docs** — if a code change affects user-facing behavior, update the relevant doc in the same PR.
8. **Local preview** — `mkdocs serve` at repo root, open http://127.0.0.1:8000.

## Gotchas carried over from Ecoki

Relevant ones only:

- **Commit docs with code, always** — #1 source of doc debt is "I'll update the docs later". Build the expectation into the review checklist.
- **`site/` out of git** — MkDocs' generated output directory. Add to `.gitignore`.
- **Relative doc paths only** — survive moves.
- **Compress images before commit** — the existing `traceml_architecture_diagram.png` is 730KB; leave it alone for MVP but flag as backlog item.
- **Live reload is a force multiplier** — document `mkdocs serve` prominently on the contributing page.

Sphinx-specific gotchas (LaTeX passes, WebP images, underline drift, `mock_heavy_deps`) do not apply.

## Out of scope — explicit backlog

These items are captured here because the project's roadmap may pick them up; the spec is the pointer.

1. **Full docstring coverage.** Today ~65% of functions documented; private helpers and some modules have none. Requires roadmap buy-in from Abhinav. Likely 1-2 phases of dedicated work per subsystem.
2. **Notes/Code_base/ integration.** CODE_MAP, CODE_WALKTHROUGH, INTEGRATION_REFERENCE, METRICS_REFERENCE, EXAMPLES_WALKTHROUGH, and ARCHITECTURE_DIAGRAMS are substantial existing content (~140KB total across six files). Integrating them transforms the Developer Guide and User Guide reference density. Post-MVP phase.
3. **Internal-only MkDocs site.** `mkdocs.internal.yml` + `docs_internal/` for TIMELINE.md and strategic notes. Same Material theme, never deployed, served locally via `mkdocs serve -f mkdocs.internal.yml`. ~30 min to add.
4. **Custom domain `docs.traceopt.ai`** via CNAME file + DNS.
5. **Versioned docs** via `mike` once tagged releases matter.
6. **PDF output** if a stakeholder demands it later.
7. **Spell-check and prose linters** (vale, codespell on prose) in CI.
8. **Architecture PNG replacement with Mermaid equivalent** — retire the 730KB PNG.

## Acceptance criteria

1. `pip install -e ".[docs]"` installs all docs dependencies cleanly.
2. `mkdocs serve` launches a site at `http://127.0.0.1:8000` with the nav tree above, no build warnings.
3. `mkdocs build --strict` exits 0 on a clean checkout.
4. User Guide pages render with all internal links resolved.
5. Public API page renders auto-generated signatures for `trace_step`, `trace_model_instance`, `TraceMLTrainer`, `TraceMLCallback`, CLI entry point.
6. Every Developer Guide subsystem page renders at least one auto-generated class or function from its module.
7. Architecture page renders the Mermaid diagram visually (not as raw code).
8. `.github/workflows/docs.yml` builds green on this branch's PR.
9. `.github/ISSUE_TEMPLATE/` templates appear in GitHub's new-issue picker after merge.
10. The PR description lists the backlog items above so Abhinav sees them at review time.

## Out-of-scope verification note

Because the existing upstream `origin/main` has diverged from `upstream/main` (observed a `ci.yml` merge conflict during initial rebase), this branch is based on `upstream/main`, not `origin/main`. Push target is the user's fork (`origin`) with `git push -u origin docs/dev-docs-site`. The eventual PR is against `upstream/main`.
