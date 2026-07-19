# TraceML RFCs

This directory holds TraceML enhancement proposals (RFCs). An RFC is a short
design document for a change that is worth discussing before it is built:
new user-facing behavior, a wire/schema change, a new integration surface, or
anything that affects backward compatibility or the core contracts (local-first,
best-effort instrumentation, low overhead).

Small, obvious fixes do not need an RFC. Open a normal issue or PR. Use an RFC
when a reviewer would reasonably ask "why this shape and not another?".

This process is intentionally lightweight. It borrows the structure of the Ray
REP, Rust RFC, and Kubernetes KEP processes, sized for a small maintainer team:
no shepherds, no voting, no separate repo. Maintainer sign-off is acceptance.

## States

```
Draft  ->  Discussion  ->  Accepted  ->  Implemented
                    \-> Rejected / Withdrawn
```

- **Draft**: proposed, still being written.
- **Discussion**: open for feedback (on the RFC pull request).
- **Accepted**: a maintainer has approved the design. Implementation may start.
- **Implemented**: the accepted design has shipped (link the release).
- **Rejected / Withdrawn**: closed without acceptance; the document stays for the
  record with the reason noted.

## How to propose an RFC

1. Copy `template.md` to `NNNN-short-title.md`, where `NNNN` is the next free
   number (zero-padded). Set `Status: Draft`.
2. Open a pull request that adds the file. The PR is the discussion thread.
3. Iterate. When a maintainer approves, flip `Status` to `Accepted` and merge.
4. Open an epic tracking issue that links the RFC and its child issues. Label the
   child issues (`good first issue`, `help wanted`, `enhancement`, ...).
5. When the work ships, set `Status: Implemented` and link the release.

## How to contribute to accepted work

Every accepted RFC lists its child issues at the bottom. Issues labeled
`good first issue` and `help wanted` are open for contributors. Comment on an
issue to claim it; see `CONTRIBUTING.md` for dev setup. Maintainers own the
correctness-critical pieces (noted per issue); the community is very welcome on
examples, docs, tests, and adapters.

## Index

| RFC | Title | Status |
|---|---|---|
| [0001](0001-surface-and-attribute-native-crashes.md) | Surface and correctly attribute native training crashes | Draft |
