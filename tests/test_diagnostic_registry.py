from traceml.diagnostics.model_diagnostics import (
    DEFAULT_MODEL_DIAGNOSTIC_REGISTRY,
    ModelDiagnosisItem,
    build_model_diagnostics_payload,
)
from traceml.diagnostics.registry import (
    DiagnosticDomainRegistry,
    DiagnosticDomainSpec,
    ModelDiagnosticContext,
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
        assert context.step_time_metrics == ()
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
        step_time_metrics=(),
        step_memory_metrics=(),
        registry=registry,
    )

    assert payload.status_message == "OK"
    assert [item.source for item in payload.items] == ["custom_domain"]
    assert payload.items[0].status == "BALANCED"


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
        step_time_metrics=(),
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
        step_time_metrics=(),
        step_memory_metrics=(),
    )

    assert [item.source for item in payload.items] == [
        "step_time",
        "step_memory",
    ]
    assert [item.status for item in payload.items] == ["NO DATA", "NO DATA"]
