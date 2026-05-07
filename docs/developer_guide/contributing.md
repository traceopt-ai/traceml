# Contributing

## Dev setup

```bash
git clone https://github.com/traceopt-ai/traceml.git
cd traceml
pip install -e ".[dev,torch,lightning,hf,docs]"
pre-commit install --install-hooks
pre-commit install --hook-type pre-push
```

All extras are installed because the docs build imports every TraceML module.

## Branch naming

- `feature/<short-name>` — new feature work.
- `fix/<short-name>` — bug fixes.
- `docs/<short-name>` — docs-only changes.

## Commit messages

Short single-line. Imperative mood. No `Co-Authored-By` trailers.

Examples:

- `feat: add step-time outlier detection`
- `fix: handle torchrun restart in rank detector`
- `docs: clarify W&B integration example`

## Code style

- `black` (line length 79), `ruff`, `isort` (black profile) — enforced by pre-commit.
- Python 3.10+ features allowed; nothing Python 3.12-only.

## Docstrings

NumPy style. Every public class, function, method gets one.

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

## Tests

```bash
pytest tests/
```

New code must include tests unless the change is docs-only.

## Docs

If a code change affects user-facing behavior, update the relevant doc in the same PR.

- **Preview locally:** `mkdocs serve`, then open http://127.0.0.1:8000
- **Before pushing:** `mkdocs build --strict` must succeed. The pre-push hook enforces this.

## Cross-referencing

Use `autorefs` to link to API symbols in prose:

```markdown
See [`trace_step`][traceml.decorators.trace_step] for the context manager.
```

## PR checklist

A populated template appears when you open a PR. Fill every box.
