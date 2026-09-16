"""Render and validate the evidence-backed integration support matrix.

The JSON manifest is the source of truth. ``--check`` is deliberately strict:
it rejects stale generated documentation, a missing example or guide, and any
``CI tested`` claim whose named CI job does not install the required extras.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any


REPOSITORY_URL = "https://github.com/traceopt-ai/traceml/blob/main/"
ISSUE_URL = "https://github.com/traceopt-ai/traceml/issues/"
REQUIRED_IDS = {
    "pytorch",
    "huggingface-trainer",
    "accelerate",
    "lightning",
    "ray-train",
    "deepspeed",
    "wandb-mlflow",
}
REQUIRED_FIELDS = {
    "id",
    "stack",
    "recommended_api",
    "guide",
    "tested_dependency_range",
    "coverage",
    "emitted_signals",
    "example",
    "validation_level",
    "limitations",
}
REQUIRED_COVERAGE = {
    "cpu",
    "gpu",
    "single_process",
    "multi_process",
    "multi_node",
}
VALIDATION_LEVELS = {
    "CI tested",
    "nightly tested",
    "documented recipe",
    "experimental",
}
# Per-scope coverage statuses. These are deliberately a different set from
# VALIDATION_LEVELS: a scope can be out of scope ("Not applicable"), untested
# ("Not claimed"), or known-broken ("Unsupported"), none of which are
# validation levels. Statuses are matched exactly, so an unlisted spelling
# such as "ci tested" must fail loudly here rather than silently bypass the
# ci_evidence requirement below.
COVERAGE_STATUSES = {
    "CI tested",
    "Documented recipe",
    "Experimental",
    "Not applicable",
    "Not claimed",
    "Unsupported",
}
REQUIRED_LIMITATIONS = {"guide", "guide_label", "issues"}
BEGIN = "<!-- BEGIN GENERATED: integration-support-matrix -->"
END = "<!-- END GENERATED: integration-support-matrix -->"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def ci_job_body(workflow: str, job: str) -> str:
    match = re.search(
        rf"^  {re.escape(job)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:|\Z)",
        workflow,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise ValueError(f"CI job {job!r} is not present in the workflow")
    return match.group("body")


def installed_extras(job_body: str) -> set[str]:
    extras: set[str] = set()
    for extra_list in re.findall(
        r"pip install -e \"\.\[([^]]+)\]\"", job_body
    ):
        extras.update(part.strip() for part in extra_list.split(","))
    return extras


def validate_manifest(
    manifest: dict[str, Any], *, root: Path, workflow_path: Path
) -> None:
    if manifest.get("schema_version") != 1:
        raise ValueError("manifest schema_version must be 1")
    if set(manifest.get("validation_levels", {})) != VALIDATION_LEVELS:
        raise ValueError(
            "manifest must define exactly the supported validation levels"
        )

    integrations = manifest.get("integrations")
    if not isinstance(integrations, list):
        raise ValueError("manifest integrations must be a list")

    ids = [entry.get("id") for entry in integrations]
    if len(ids) != len(set(ids)):
        raise ValueError("integration IDs must be unique")
    if set(ids) != REQUIRED_IDS:
        missing = sorted(REQUIRED_IDS - set(ids))
        unexpected = sorted(set(ids) - REQUIRED_IDS)
        raise ValueError(
            f"manifest integration set changed; missing={missing}, unexpected={unexpected}"
        )

    workflow = workflow_path.read_text(encoding="utf-8")
    for entry in integrations:
        missing_fields = REQUIRED_FIELDS - set(entry)
        if missing_fields:
            raise ValueError(
                f"{entry['id']}: missing fields {sorted(missing_fields)}"
            )
        if entry["validation_level"] not in VALIDATION_LEVELS:
            raise ValueError(f"{entry['id']}: invalid validation level")
        if not (root / "docs" / entry["guide"].split("#", 1)[0]).is_file():
            raise ValueError(f"{entry['id']}: missing guide {entry['guide']}")
        if not (root / entry["example"]).is_file():
            raise ValueError(
                f"{entry['id']}: missing example {entry['example']}"
            )

        coverage = entry["coverage"]
        if set(coverage) != REQUIRED_COVERAGE:
            raise ValueError(
                f"{entry['id']}: coverage must define every matrix scope"
            )
        for scope, claim in coverage.items():
            if not isinstance(claim, dict) or not {"status", "detail"} <= set(
                claim
            ):
                raise ValueError(
                    f"{entry['id']}: {scope} has an invalid coverage claim"
                )
            if claim["status"] not in COVERAGE_STATUSES:
                raise ValueError(
                    f"{entry['id']}: {scope} has unknown coverage status "
                    f"{claim['status']!r}; expected one of "
                    f"{sorted(COVERAGE_STATUSES)}"
                )

        limitations = entry["limitations"]
        if not isinstance(limitations, dict):
            raise ValueError(f"{entry['id']}: limitations must be an object")
        missing_limitations = REQUIRED_LIMITATIONS - set(limitations)
        if missing_limitations:
            raise ValueError(
                f"{entry['id']}: limitations missing "
                f"{sorted(missing_limitations)}"
            )
        limitations_guide = limitations["guide"].split("#", 1)[0]
        if not (root / "docs" / limitations_guide).is_file():
            raise ValueError(
                f"{entry['id']}: missing limitations guide "
                f"{limitations['guide']}"
            )
        if not isinstance(limitations["issues"], list):
            raise ValueError(
                f"{entry['id']}: limitations issues must be a list"
            )

        is_ci_claim = entry["validation_level"] == "CI tested" or any(
            claim["status"] == "CI tested" for claim in coverage.values()
        )
        evidence = entry.get("ci_evidence")
        if is_ci_claim and not evidence:
            raise ValueError(
                f"{entry['id']}: CI tested claims require ci_evidence"
            )
        if not evidence:
            continue

        for field in ("job", "extras", "test", "label"):
            if field not in evidence:
                raise ValueError(f"{entry['id']}: ci_evidence missing {field}")
        if not (root / evidence["test"]).is_file():
            raise ValueError(
                f"{entry['id']}: missing cited test {evidence['test']}"
            )
        job_body = ci_job_body(workflow, evidence["job"])
        # Existence on disk is not evidence that CI runs it. The job invokes
        # pytest with directories rather than individual files, so accept the
        # cited path or any parent directory of it. Each candidate must appear
        # as a whole path token: a bare "tests" must not match "tests/renderers"
        # and count an unrelated suite as evidence.
        test_path = PurePosixPath(evidence["test"])
        invoked = [test_path, *test_path.parents]
        # Search only the shell commands. YAML keys and comments are prose:
        # a step named "Run integration tests" or a comment mentioning
        # "timing tests" must not satisfy the bare "tests" parent candidate.
        commands = "\n".join(
            line.split("#", 1)[0]
            for line in job_body.splitlines()
            if not re.match(r"\s*-?\s*[A-Za-z_][\w-]*:", line)
        )
        if not any(
            re.search(
                rf"(?<![\w/]){re.escape(str(candidate))}(?![\w/])", commands
            )
            for candidate in invoked
            if str(candidate) != "."
        ):
            raise ValueError(
                f"{entry['id']}: CI job {evidence['job']!r} does not invoke "
                f"cited test {evidence['test']} or a directory containing it"
            )
        job_extras = installed_extras(job_body)
        missing_extras = set(evidence["extras"]) - job_extras
        if missing_extras:
            raise ValueError(
                f"{entry['id']}: CI job {evidence['job']!r} does not install "
                f"claimed extras {sorted(missing_extras)}"
            )


def source_link(path: str, label: str) -> str:
    return f"[{label}]({REPOSITORY_URL}{path})"


def scope_claim(claim: dict[str, str]) -> str:
    return f"**{claim['status']}** — {claim['detail']}"


def guide_link(path: str, label: str) -> str:
    # The generated table lives in docs/user_guide/, while manifest paths are
    # rooted at docs/. Keep the manifest useful for file-existence validation
    # without emitting broken documentation links.
    relative_path = path.replace("user_guide/", "", 1)
    return f"[{label}]({relative_path})"


def limitations_cell(limitations: dict[str, Any]) -> str:
    links = [guide_link(limitations["guide"], limitations["guide_label"])]
    links.extend(
        f"[#{issue}]({ISSUE_URL}{issue})" for issue in limitations["issues"]
    )
    return " · ".join(links)


def render_matrix(manifest: dict[str, Any]) -> str:
    rows = [
        "## Integration support matrix",
        "",
        "The manifest at `docs/data/integration_support.json` is the source "
        "of truth for this table. A status is evidence, not a promise: `CI "
        "tested` means the linked job installs the extra and runs the linked "
        "real-framework test. `documented recipe` and `experimental` are not "
        "end-to-end validation.",
        "",
        "| Stack | Recommended API | Tested dependency range | CPU / GPU | Single- / multi-process | Multi-node | Emitted signals | Example | Validation level | Limitations |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for entry in manifest["integrations"]:
        coverage = entry["coverage"]
        device = f"CPU: {scope_claim(coverage['cpu'])}<br>GPU: {scope_claim(coverage['gpu'])}"
        process = (
            f"Single: {scope_claim(coverage['single_process'])}<br>"
            f"Multi: {scope_claim(coverage['multi_process'])}"
        )
        example = source_link(
            entry["example"], entry["example"].rsplit("/", 1)[-1]
        )
        validation = f"**{entry['validation_level']}**"
        evidence = entry.get("ci_evidence")
        if evidence:
            validation += (
                "<br>"
                + source_link(
                    ".github/workflows/ci.yml", f"CI job: {evidence['job']}"
                )
                + "; "
                + source_link(evidence["test"], evidence["label"])
            )
        guide = guide_link(entry["guide"], entry["stack"])
        cells = [
            guide,
            f"`{entry['recommended_api']}`",
            entry["tested_dependency_range"],
            device,
            process,
            scope_claim(coverage["multi_node"]),
            entry["emitted_signals"],
            example,
            validation,
            limitations_cell(entry["limitations"]),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    rows.extend(
        [
            "",
            "### Evidence limits",
            "",
            "- This matrix makes no GPU, multi-process, or multi-node claim "
            "without a corresponding reproducible job.",
            "- Lightning, Ray, and DeepSpeed tests in the current integration "
            "suite include mocked, lazy-import, or skipped paths; they are "
            "therefore not classified as end-to-end validation here.",
            "- Run `python tools/integration_support_matrix.py --check` after "
            "editing the manifest, examples, or CI install extras. The check "
            "fails if a `CI tested` row loses its cited extra, test, guide, "
            "example, or generated documentation row.",
        ]
    )
    return "\n".join(rows)


def generated_document(contents: str, matrix: str) -> str:
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)
    replacement = f"{BEGIN}\n{matrix}\n{END}"
    if not pattern.search(contents):
        raise ValueError(
            "documentation is missing the generated matrix markers"
        )
    return pattern.sub(replacement, contents, count=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="validate without writing"
    )
    parser.add_argument(
        "--write", action="store_true", help="write the generated matrix"
    )
    args = parser.parse_args()
    if args.check and args.write:
        parser.error("choose either --check or --write")

    root = repository_root()
    manifest_path = root / "docs/data/integration_support.json"
    documentation_path = root / "docs/user_guide/integrations.md"
    workflow_path = root / ".github/workflows/ci.yml"
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest, root=root, workflow_path=workflow_path)
    rendered = generated_document(
        documentation_path.read_text(encoding="utf-8"), render_matrix(manifest)
    )

    if args.write:
        documentation_path.write_text(rendered, encoding="utf-8")
    elif documentation_path.read_text(encoding="utf-8") != rendered:
        raise ValueError(
            "integration matrix is stale; run "
            "python tools/integration_support_matrix.py --write"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(
            f"integration support matrix check failed: {error}",
            file=sys.stderr,
        )
        raise SystemExit(1)
