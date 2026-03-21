# CLAUDE.md

Project-level instructions for Claude Code. Auto-loaded every conversation.

## Project

TraceML (`traceml-ai` on PyPI) -- lightweight, real-time bottleneck finder for PyTorch training runs.
Built by traceopt.ai (founder: Abhinav Srivastav, abhinav@traceopt.ai).

## Repository layout

- `src/traceml/` -- main package
  - `cli.py` -- CLI entry point (`traceml watch|run|deep`)
  - `decorators.py` -- core API (`trace_step`, `trace_model_instance`)
  - `aggregator/` -- out-of-process telemetry server, summaries
  - `runtime/` -- per-rank in-process agent
  - `samplers/` -- telemetry collectors (step time, memory, system, layers)
  - `database/` -- in-memory + SQLite telemetry storage
  - `transport/` -- TCP between ranks and aggregator
  - `integrations/` -- HuggingFace Trainer, PyTorch Lightning
- `src/examples/` -- runnable example scripts
- `tests/` -- pytest test suite
- `docs/` -- quickstart, integration guides

## Dev setup

```bash
pip install -e ".[dev,torch]"
```

## Conventions

- Keep commit messages short, single line. No `Co-Authored-By` trailers.
- Python 3.10+. Line length 79 (black/ruff).
- Use `py` launcher on Windows, `python3` on Linux/Mac.
- Package source lives in `src/` (setuptools `package-dir = {"" = "src"}`).
