# Security Policy

TraceML runs inside ML training jobs and collects local runtime telemetry. Security
reports are important to us, especially when they involve training environments,
distributed launch paths, local report artifacts, or dependency behavior.

## Supported Versions

Security fixes are applied to the latest released version of `traceml-ai`.

| Version | Supported |
| --- | --- |
| Latest PyPI release | Yes |
| Older releases | Best effort |
| Unreleased `main` branch | Best effort |

If you are unsure whether a finding affects a supported version, report it
anyway and include the TraceML version or commit SHA you tested.

## Reporting a Vulnerability

Please do not open a public GitHub issue for a suspected vulnerability.

Report security issues by email:

```text
support@traceopt.ai
```

Use the subject:

```text
[TraceML Security] <short description>
```

If GitHub private vulnerability reporting is enabled for this repository, you
may use that instead.

## What To Include

Useful reports include:

- affected TraceML version or commit SHA
- operating system, Python, PyTorch, CUDA, and launcher details if relevant
- whether the issue affects `traceml run`, `traceml watch`, `traceml compare`,
  dashboard mode, integrations, or generated artifacts
- minimal reproduction steps or proof of concept
- expected impact, such as data exposure, command execution, path traversal,
  unsafe network binding, dependency confusion, denial of service, or privilege
  boundary concerns
- whether the issue is already public

Please remove secrets, tokens, private dataset paths, customer names, cluster
hostnames, and proprietary model details from logs and reports before sending
them unless they are necessary to demonstrate the issue.

## Scope

We are especially interested in reports involving:

- unintended exposure of training metadata, logs, hostnames, paths, or
  environment details
- unsafe handling of `final_summary.json`, compare artifacts, SQLite telemetry,
  or local dashboard inputs
- command injection, path traversal, unsafe file writes, or unsafe process
  launching in the TraceML CLI
- unsafe network behavior in the aggregator, worker telemetry transport, or
  distributed launch configuration
- vulnerabilities in required runtime dependencies that affect TraceML users
- cases where TraceML can crash, hang, or materially interfere with a user's
  training job because of malformed telemetry or report input

Out of scope:

- vulnerabilities in a user's training script, model code, dataset, cluster
  scheduler, CUDA driver, PyTorch installation, or experiment tracker unless
  TraceML materially enables or worsens the issue
- reports that rely only on local machine access without a TraceML-specific
  privilege boundary impact
- social engineering, spam, or denial-of-service against project maintainers

## Response Expectations

We aim to acknowledge valid security reports within 2 business days.

After acknowledgement, we will:

- confirm the affected versions and severity
- ask for more detail if the report is not reproducible
- coordinate a fix and release when needed
- credit the reporter if desired
- publish a public advisory or release note when appropriate

Timelines depend on severity and complexity, but we will keep reporters updated
while a fix is in progress.

## Disclosure

Please give us a reasonable opportunity to investigate and release a fix before
public disclosure. We prefer coordinated disclosure and will work with reporters
on timing when users need to take action.

## Security Design Notes

TraceML is designed to be local-first and low overhead. By default, it writes
local artifacts such as:

```text
logs/<run_name>/final_summary.json
logs/<run_name>/final_summary.txt
```

These artifacts can contain operational metadata from the training environment.
Before sharing them publicly, review and redact sensitive fields such as
hostnames, usernames, paths, cluster names, dataset names, and internal run
identifiers.

For multi-node runs, TraceML starts an aggregator on the owner node and workers
send telemetry to it. Only bind the aggregator to interfaces reachable by the
intended workers, and prefer trusted networks for distributed training traffic.
