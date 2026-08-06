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
    expected_install = 'pip install -e ".[torch,dashboard,hf]"'
    assert expected_install in original
    workflow.write_text(
        original.replace(
            expected_install, 'pip install -e ".[torch,dashboard]"'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError, match="does not install claimed extras.*hf"
    ):
        module.validate_manifest(manifest, root=ROOT, workflow_path=workflow)
