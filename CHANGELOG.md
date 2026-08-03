# Changelog

All notable changes to TraceML are documented here. This file follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions here
should match the tags on [GitHub Releases](https://github.com/traceopt-ai/traceml/releases),
which carry the full historical notes for versions predating this file.

## [Unreleased]

- Step Time now has one typed `load -> analyze -> diagnose -> present`
  pipeline. CLI and dashboard share the live profile, the dashboard fans one
  analysis to both views, and final summary runs the summary profile once
  before a pure JSON/text projection.
- Step Time SQLite reads are set-based and rank-independent, and unchanged
  live source cursors reuse the existing analysis instead of decoding and
  diagnosing the same window again.
- **Breaking:** `traceml_ai.utils.step_time_window` and
  `traceml_ai.utils.step_time_sqlite` were removed. Use
  `traceml_ai.step_time.model`, `traceml_ai.step_time.sqlite`,
  `traceml_ai.step_time.analysis`, and `traceml_ai.step_time.pipeline`.
- `build_step_diagnosis()`, `build_step_diagnosis_result()`, and the
  reporting-only `RankStepSummary` remain available as deprecated one-release
  compatibility shims. New integrations should pass a canonical
  `StepTimeWindow` to `diagnose_step_time_window(window, policy=...)` and
  project `StepTimeRankFacts` / `StepTimeValues` directly.
- NumPy 2 is supported: the `numpy<2` cap is removed, so installing
  TraceML no longer downgrades NumPy in a modern environment and
  resolvers no longer fall back to old TraceML releases.
- The aggregator starts on Windows: `SO_REUSEPORT` is applied only on
  platforms that have it.
- `plotly` is no longer a dependency; nothing imported it, and its
  absence no longer blocks dashboard mode.
- `nicegui` carries a version floor covering the dashboard's API use,
  so resolvers can no longer select a version that breaks at runtime.
- Runtime dependencies are guarded against undocumented upper bounds,
  and a scheduled CI leg resolves the newest published dependencies.
- CONTRIBUTING documents the dependency policy behind these changes.
- Cutting a release now also creates the GitHub Release entry, so a
  published version is never missing its release notes.
- The release workflow accepts four-component patch versions
  (for example 0.3.5.1).

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
