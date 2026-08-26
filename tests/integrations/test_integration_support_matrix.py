"""Regression tests for the generated integration support matrix."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "tools/integration_support_matrix.py"
MANIFEST = ROOT / "docs/data/integration_support.json"
WORKFLOW = ROOT / ".github/workflows/ci.yml"


def _support_matrix_module():
    spec = importlib.util.spec_from_file_location("support_matrix", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_support_matrix_rejects_a_missing_manifest_entry():
    module = _support_matrix_module()
    manifest = module.load_manifest(MANIFEST)
    manifest["integrations"] = [
        entry
        for entry in manifest["integrations"]
        if entry["id"] != "ray-train"
    ]

    with pytest.raises(ValueError, match="missing=.*ray-train"):
        module.validate_manifest(manifest, root=ROOT, workflow_path=WORKFLOW)


def test_support_matrix_rejects_a_ci_claim_when_its_extra_is_removed(tmp_path):
    module = _support_matrix_module()
    manifest = module.load_manifest(MANIFEST)
    workflow = tmp_path / "ci.yml"
    original = WORKFLOW.read_text(encoding="utf-8")
    expected_install = 'pip install -e ".[torch,dashboard,hf,otlp]"'
    assert expected_install in original
    workflow.write_text(
        original.replace(
            expected_install, 'pip install -e ".[torch,dashboard,otlp]"'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="does not install claimed extras.*hf"
    ):
        module.validate_manifest(manifest, root=ROOT, workflow_path=workflow)


def test_committed_manifest_validates_and_matches_the_generated_docs():
    module = _support_matrix_module()
    manifest = module.load_manifest(MANIFEST)

    module.validate_manifest(manifest, root=ROOT, workflow_path=WORKFLOW)

    documentation = (ROOT / "docs/user_guide/integrations.md").read_text(
        encoding="utf-8"
    )
    assert documentation == module.generated_document(
        documentation, module.render_matrix(manifest)
    ), "integrations.md is stale; run tools/integration_support_matrix.py --write"


def test_support_matrix_rejects_an_unknown_coverage_status():
    module = _support_matrix_module()
    manifest = module.load_manifest(MANIFEST)
    # A near-miss spelling must fail loudly instead of silently bypassing the
    # ci_evidence requirement, which keys off the exact string "CI tested".
    manifest["integrations"][0]["coverage"]["cpu"]["status"] = "ci tested"

    with pytest.raises(ValueError, match=r"unknown coverage status"):
        module.validate_manifest(manifest, root=ROOT, workflow_path=WORKFLOW)


def test_support_matrix_rejects_incomplete_limitations():
    module = _support_matrix_module()
    manifest = module.load_manifest(MANIFEST)
    # Previously this passed validation and then raised KeyError while
    # rendering the limitations cell.
    manifest["integrations"][0]["limitations"].pop("issues")

    with pytest.raises(ValueError, match=r"limitations missing.*issues"):
        module.validate_manifest(manifest, root=ROOT, workflow_path=WORKFLOW)


def test_support_matrix_rejects_a_cited_test_the_job_never_runs():
    module = _support_matrix_module()
    manifest = module.load_manifest(MANIFEST)
    entry = next(
        item for item in manifest["integrations"] if item.get("ci_evidence")
    )
    # The file exists, but the cited job does not run it or its directory.
    entry["ci_evidence"]["test"] = "tests/config/test_yaml_loader.py"
    assert (ROOT / entry["ci_evidence"]["test"]).is_file()

    with pytest.raises(ValueError, match=r"does not invoke cited test"):
        module.validate_manifest(manifest, root=ROOT, workflow_path=WORKFLOW)
