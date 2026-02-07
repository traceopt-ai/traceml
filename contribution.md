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
- Python 3.9+
- PyTorch 1.12+
- CUDA enabled GPU (recommended; contributors may use local GPUs or cloud providers such as Google Cloud, AWS, Runpod, etc.)

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

---

## Pull Request Guidelines

- Keep PRs focused and small when possible
- Add comments explaining *why*, not just *what*
- Avoid breaking existing APIs unless discussed
- Run basic training loops to verify no regressions
- Performance-sensitive changes should include a short explanation

---

## What We Are Not Looking For (Yet)

To keep the project focused, we are currently **not** accepting:

- Large refactors without prior discussion
- New heavy dependencies
- Framework rewrites or abstraction layers
- Features that significantly increase runtime overhead

---

## Communication

- Bugs and features: GitHub Issues
- Design discussions: GitHub Discussions
- Security issues: email abhinav@traceopt.ai

Maintainer: Abhinav Srivastav

---

Thanks for helping make TraceML better 🚀
