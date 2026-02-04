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

## Contributor License Agreement

By submitting a pull request, you agree that you grant the project maintainer
a perpetual, irrevocable, worldwide, royalty-free license to use, modify,
distribute, and sublicense your contribution as part of the TraceML project.

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
- CUDA enabled GPU (recommended; contributors may use local GPUs or cloud providers such Google, AWS, Runpod, etc.)

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
