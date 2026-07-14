from types import SimpleNamespace

from traceml_ai.diagnostics.step_memory import LIVE_STEP_MEMORY_POLICY
from traceml_ai.diagnostics.model_diagnostics import (
    DEFAULT_MODEL_DIAGNOSTIC_REGISTRY,
    ModelDiagnosisItem,
    build_model_diagnostics_payload,
)
from traceml_ai.diagnostics.registry import (
    DiagnosticDomainRegistry,
    DiagnosticDomainSpec,
    ModelDiagnosticContext,
)
from traceml_ai.renderers.step_memory.schema import (
    StepMemoryCombinedCoverage,
    StepMemoryCombinedMetric,
    StepMemoryCombinedSeries,
    StepMemoryCombinedSummary,
)
from traceml_ai.renderers.step_time.schema import (
    StepCombinedTimeCoverage,
    StepCombinedTimeMetric,
    StepCombinedTimeSummary,
)


def _step_memory_metric() -> StepMemoryCombinedMetric:
    return StepMemoryCombinedMetric(
        metric="peak_reserved",
        device="cuda:0",
        series=StepMemoryCombinedSeries(
            steps=[1, 2, 3],
            median=[100.0, 100.0, 100.0],
            worst=[100.0, 100.0, 100.0],
        ),
        summary=StepMemoryCombinedSummary(
            window_size=3,
            steps_used=3,
            median_peak=100.0,
            worst_peak=100.0,
            worst_rank=0,
            skew_ratio=0.0,
            skew_pct=0.0,
        ),
        coverage=StepMemoryCombinedCoverage(
            expected_steps=3,
            steps_used=3,
            completed_step=3,
            world_size=1,
            ranks_present=1,
            incomplete=False,
        ),
    )


def _step_time_metric(name: str, value: float) -> StepCombinedTimeMetric:
    return StepCombinedTimeMetric(
        metric=name,
        clock="gpu",
        series=None,
        summary=StepCombinedTimeSummary(
            window_size=1,
            steps_used=1,
            median_total=value,
            worst_total=value,
            worst_rank=0,
            skew_ratio=0.0,
            skew_pct=0.0,
        ),
        coverage=StepCombinedTimeCoverage(
            expected_steps=1,
            steps_used=1,
            completed_step=1,
            world_size=1,
            ranks_present=1,
            incomplete=False,
        ),
    )


def test_default_model_diagnostic_registry_contains_primary_domains():
    assert DEFAULT_MODEL_DIAGNOSTIC_REGISTRY.keys() == (
        "step_time",
        "step_memory",
    )


def test_model_diagnostics_payload_uses_registered_domains():
    def build_custom_item(
        context: ModelDiagnosticContext,
    ) -> ModelDiagnosisItem:
        assert context.step_time_per_rank_timing == {}
        assert context.step_memory_metrics == ()
        return ModelDiagnosisItem(
            source="custom_domain",
            title="Custom Domain",
            kind="BALANCED",
            severity="info",
            status="BALANCED",
            reason="Custom diagnostic is healthy.",
            action="Keep monitoring.",
        )

    registry = DiagnosticDomainRegistry(
        (
            (
                "custom_domain",
                DiagnosticDomainSpec(
                    name="custom_domain",
                    title="Custom Domain",
                    builder=build_custom_item,
                ),
            ),
        )
    )

    payload = build_model_diagnostics_payload(
        step_memory_metrics=(),
        registry=registry,
    )

    assert payload.status_message == "OK"
    assert [item.source for item in payload.items] == ["custom_domain"]
    assert payload.items[0].status == "BALANCED"


def test_model_step_time_diagnostics_receive_per_rank_timing(monkeypatch):
    per_rank_timing = {
        0: {"input_wait": 1.0, "total_step": 10.0},
        1: {"input_wait": 2.0, "total_step": 11.0},
    }
    captured = {}

    def fake_diagnosis(metrics, *, per_rank_timing=None, **kwargs):
        captured["metrics"] = metrics
        captured["per_rank_timing"] = per_rank_timing
        return SimpleNamespace(
            kind="BALANCED",
            severity="info",
            status="BALANCED",
            reason="No timing issue.",
            action="Keep monitoring.",
            note=None,
            confidence=0.75,
            steps_used=3,
            worst_rank=1,
        )

    monkeypatch.setattr(
        model_diagnostics,
        "build_step_diagnosis",
        fake_diagnosis,
    )

    payload = build_model_diagnostics_payload(
        step_time_per_rank_timing=per_rank_timing,
        step_memory_metrics=(),
    )

    assert captured["per_rank_timing"] == per_rank_timing
    assert payload.items[0].source == "step_time"
    assert payload.items[0].status == "BALANCED"


def test_model_step_time_diagnostics_use_selected_metrics(monkeypatch):
    diagnosis_metrics = (
        _step_time_metric("input_wait", 40.0),
        _step_time_metric("step_time", 100.0),
        _step_time_metric("residual_proxy", 0.0),
    )
    captured = {}

    def fake_diagnosis(metrics, **kwargs):
        captured["metrics"] = metrics
        captured["diagnosis_clock"] = kwargs.get("diagnosis_clock")
        return SimpleNamespace(
            kind="INPUT_BOUND",
            severity="warn",
            status="INPUT-BOUND",
            reason="Input wait is high.",
            action="Increase workers.",
            note=None,
            confidence=0.75,
            steps_used=1,
            worst_rank=0,
        )

    monkeypatch.setattr(
        model_diagnostics,
        "build_step_diagnosis",
        fake_diagnosis,
    )

    payload = build_model_diagnostics_payload(
        step_time_diagnosis_metrics=diagnosis_metrics,
        step_time_diagnosis_clock="gpu",
        step_memory_metrics=(),
    )

    assert captured["metrics"] == diagnosis_metrics
    assert captured["diagnosis_clock"] == "gpu"
    assert payload.items[0].status == "INPUT-BOUND"
    assert payload.items[0].evidence["dominant"] == "input wait"


def test_model_step_time_diagnostics_do_not_fallback_to_public_metrics(
    monkeypatch,
):
    captured = {}

    def fake_diagnosis(metrics, **kwargs):
        captured["metrics"] = metrics
        return SimpleNamespace(
            kind="NO_DATA",
            severity="info",
            status="NO DATA",
            reason="No selected diagnosis metrics.",
            action="Wait for data.",
            note=None,
            confidence=None,
            steps_used=0,
            worst_rank=None,
        )

    monkeypatch.setattr(
        model_diagnostics,
        "build_step_diagnosis",
        fake_diagnosis,
    )

    payload = build_model_diagnostics_payload(
        step_time_diagnosis_metrics=(),
        step_memory_metrics=(),
    )

    assert captured["metrics"] == ()
    assert payload.items[0].evidence == {}


def test_model_diagnostics_domain_failures_are_fallback_items():
    def broken_builder(context: ModelDiagnosticContext) -> ModelDiagnosisItem:
        raise RuntimeError("boom")

    registry = DiagnosticDomainRegistry(
        (
            (
                "unstable",
                DiagnosticDomainSpec(
                    name="unstable",
                    title="Unstable",
                    builder=broken_builder,
                ),
            ),
        )
    )

    payload = build_model_diagnostics_payload(
        step_memory_metrics=(),
        registry=registry,
    )

    assert payload.status_message == "OK"
    assert payload.items[0].source == "unstable"
    assert payload.items[0].status == "NO DATA"
    assert (
        payload.items[0].reason
        == "Unstable diagnosis is unavailable on this tick."
    )


def test_default_model_diagnostics_payload_keeps_existing_sources():
    payload = build_model_diagnostics_payload(
        step_memory_metrics=(),
    )

    assert [item.source for item in payload.items] == [
        "step_time",
        "step_memory",
    ]
    assert [item.status for item in payload.items] == ["NO DATA", "NO DATA"]


def test_model_step_memory_diagnostics_use_live_policy(monkeypatch):
    import traceml_ai.diagnostics.model_diagnostics as model_diagnostics

    captured = {}

    def fake_diagnosis(metrics, *, thresholds, **kwargs):
        captured["thresholds"] = thresholds
        return SimpleNamespace(
            kind="BALANCED",
            severity="info",
            status="BALANCED",
            reason="No memory issue.",
            action="Keep monitoring.",
            note=None,
            confidence=0.75,
            steps_used=3,
            worst_rank=0,
        )

    monkeypatch.setattr(
        model_diagnostics,
        "build_step_memory_diagnosis",
        fake_diagnosis,
    )

    payload = build_model_diagnostics_payload(
        step_memory_metrics=(_step_memory_metric(),),
    )

    assert captured["thresholds"] is LIVE_STEP_MEMORY_POLICY.thresholds
    assert payload.items[1].source == "step_memory"
    assert payload.items[1].status == "BALANCED"
