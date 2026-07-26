# Changelog

All notable changes to TraceML are documented here. This file follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions here
should match the tags on [GitHub Releases](https://github.com/traceopt-ai/traceml/releases),
which carry the full historical notes for versions predating this file.

## [Unreleased]

- Cutting a release now also creates the GitHub Release entry, so a
  published version is never missing its release notes.

## [0.3.5] - 2026-07-26

- DeepSpeed training is supported alongside DDP and FSDP.
- Telemetry export moved to a dedicated exporter thread, keeping TCP
  sends off the training thread's critical path.
- Runtime metadata (training strategy, environment) is captured and
  carried into the final summary.
- Step Time diagnosis refinements across straggler, input-bound, and
  H2D paths, including FSDP warm-step filtering.
- Package version is now derived from the git tag (`setuptools-scm`)
  instead of a hand-edited string in `pyproject.toml`.
- Releases publish to PyPI automatically on a `v*` tag push, via PyPI
  Trusted Publishing (OIDC, no stored token).
- CI now smoke-tests a clean install of the built wheel against the
  documented CLI surface.
