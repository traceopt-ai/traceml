# Contributing to TraceML

Thanks for your interest in contributing to TraceML!
We welcome contributions from the community, especially around performance, robustness, and usability.

## Ways to Contribute

You can help in many ways:

- ⭐ Starring the repository
- 🐛 Reporting bugs or unexpected behavior
- 💡 Suggesting features or diagnostics
- 🔧 Submitting pull requests
- 📖 Improving documentation or examples

If you are unsure where to start, check the GitHub Issues page or open a discussion.
Look for issues labeled `good first issue` or `help wanted` when you want a
scoped task.

---

## License of Contributions (Apache-2.0)

TraceML is licensed under the **Apache License 2.0** (see `LICENSE`).

By submitting a pull request, you agree that your contribution is intentionally
submitted for inclusion in the project and is provided under the terms of the
Apache License 2.0 (including Section 5: “Submission of Contributions”), with
no additional terms or conditions.

You also represent that you have the right to submit the contribution (e.g., it
is your original work or you are authorized to submit it, and it is not subject
to conflicting employer or third-party IP obligations).

All contributions must be submitted via GitHub pull requests.

---

## Development Setup

```bash
git clone https://github.com/traceopt-ai/traceml.git
cd traceml
pip install -e ".[dev]"
```

Requirements:
- Python 3.10+
- PyTorch 2.5+ for Torch-backed examples and tests
- CUDA enabled GPU for GPU/distributed changes. Many docs, reporting, CLI, and CPU smoke-test changes can be developed without a GPU.

### Develop in a container

If you use VS Code with the Dev Containers extension (or GitHub Codespaces),
open the repository in the provided dev container. It installs an editable
`.[dev,torch]` build and enables the pre-commit hooks for you, so the
environment matches what CI expects. The container is CPU-only; GPU work still
needs a local CUDA setup.

---

## Open tasks for first contributors

These are real starter-sized tasks we are willing to review. Please comment on
the matching GitHub issue before starting, or open a short issue first if one
does not exist yet.

### Add an end-to-end final-summary smoke test

Add a pytest test that runs a tiny TraceML training script through `traceml run`
with `--mode=summary` and a fixed `--run-name`.

Done means:

- the test runs on CPU in CI
- it asserts `logs/<run-name>/final_summary.json` is written
- it also checks `final_summary.txt` is written
- the JSON includes `schema_version`, `system`, `process`, `step_time`, and `step_memory`

### Add an executable MLflow summary example

Add a minimal example that logs `traceml.summary()` output to MLflow.

Done means:

- the example lives under `examples/`
- it keeps `mlflow` optional and does not add it as a core dependency
- it logs numeric summary values with `mlflow.log_metrics(...)`
- it logs string diagnosis fields with `mlflow.set_tags(...)`
- `examples/README.md` links to the new example

### Improve runtime-failure safety tests

TraceML should not stop user training when telemetry or reporting fails. Add or
extend tests around that contract.

Done means:

- the test simulates a TraceML runtime, sampler, sender, or final-summary failure
- the user training loop still finishes
- the failure is logged or handled without escaping into user code

### Add compare regression coverage for diagnosis changes

`traceml compare` should clearly show when a diagnosis changes between two
saved `final_summary.json` files.

Done means:

- the test uses small summary fixtures, not a full training run
- it covers at least one improvement and one regression case
- it asserts the compact text output includes the changed Step Time diagnosis
- it asserts the structured compare JSON records the same change

---

## Design Principles (Very Important)

TraceML is designed to be:

- **Low overhead** (always-on tracing)
- **Non-blocking** (no global synchronization)
- **Safe** (never crash user training loops)
- **Explicit** (no magic instrumentation)

When contributing, please ensure:

- No unnecessary CUDA synchronizations
- No blocking I/O on the training path
- Tracing code must fail gracefully
- Overhead is justified and measurable

If in doubt, open an issue before implementing.

For code extension points, see
[`docs/developer_guide/extending.md`](docs/developer_guide/extending.md).

---

## Pull Request Guidelines

- Keep PRs focused and small when possible
- Add comments explaining *why*, not just *what*
- Avoid breaking existing APIs unless discussed
- Run basic training loops to verify no regressions
- Performance-sensitive changes should include a short explanation

CodeRabbit reviews every pull request automatically and leaves inline comments.
Treat it like a first-pass reviewer: worth reading, not a blocker. A maintainer
still reviews and merges every PR.

---

## What We Are Not Looking For (Yet)

To keep the project focused, we are currently **not** accepting:

- Large refactors without prior discussion
- New heavy dependencies
- Framework rewrites or abstraction layers
- Features that significantly increase runtime overhead

---

## Dependency Policy

`pyproject.toml` is user-facing API. A wrong constraint does not break our
CI; it breaks a stranger's environment, silently. A `numpy<2` cap shipped
this way once: CI stayed green while `pip install traceml-ai` downgraded
NumPy 2 in place, and resolvers that refuse to downgrade backsolved users
to a months-old release instead (#263). These rules exist so that class of
defect cannot ship again.

1. **No upper bounds without a recorded reason.** A cap on a runtime
   dependency is only allowed for a concrete, current incompatibility,
   recorded in `ALLOWED_UPPER_BOUNDS` in
   `tests/core/test_packaging_constraints.py`. The test fails on any
   undocumented cap. "Might break someday" is not a reason: users can pin
   around a missing cap, but cannot relax one we ship.
2. **Lower bounds follow the APIs we call.** When code depends on a
   specific API of a dependency, the dependency carries a floor naming the
   first version that has it (e.g. `nicegui>=1.4.23` for `ui.context`),
   with a comment in `pyproject.toml` recording why. Without a floor, a
   resolver working around an unrelated conflict can select a version that
   imports fine and breaks at runtime.
3. **If the core import path does not import it, it is an extra.** Runtime
   dependencies are for code every user runs. Anything only some modes or
   integrations need belongs in an optional extra. The dashboard stack is
   the known exception today; whether it migrates to an extra is tracked
   in #268.
4. **CI hears ecosystem drift before users do.** A scheduled CI leg
   resolves the newest published versions of all dependencies and runs the
   core suite, so a new NumPy or NiceGUI that breaks us is our alarm
   rather than a user's bug report.

Background reading:
[Should You Use Upper Bound Version Constraints?](https://iscinumpy.dev/post/bound-version-constraints/)
(Henry Schreiner) on why caps are harmful in Python's flat dependency
model, and [SPEC 0](https://scientific-python.org/specs/spec-0000/) for
the support windows the scientific Python ecosystem expects.

---

## Releasing (maintainers)

There is no version to edit anywhere in the repo. The package version comes
from the git tag (`setuptools-scm`), and a tag is what starts a release.

**Cut a release from the Actions tab:** go to Actions -> "Cut a release" ->
Run workflow, type the version (e.g. `0.3.5`, no leading `v`). This tags,
builds, and publishes to PyPI in one run. No git commands to remember.

Pushing a tag from a local checkout still works too, if you'd rather:

```bash
git tag v0.3.5
git push upstream v0.3.5
```

Either path publishes to PyPI automatically. From there, conda-forge's own
bot picks up the new PyPI release and opens a version-bump PR on the
`traceml-ai` feedstock. If the release only changed the version, that PR
merges itself. If it added, dropped, or repinned a runtime dependency,
someone edits the recipe in that bot PR by hand before it merges, since the
bot only knows how to bump a version number.

---

## Communication

- Bugs and features: GitHub Issues
- Design discussions: GitHub Discussions
- Security issues: email support@traceopt.ai

Maintainer: OptAI UG (haftungsbeschränkt)

---

Thanks for helping make TraceML better 🚀
