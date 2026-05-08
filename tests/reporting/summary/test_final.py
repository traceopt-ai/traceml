from dataclasses import dataclass

from traceml.core.summaries import SummaryResult
from traceml.reporting.final import (
    FinalReportGenerator,
    build_summary_payload,
)


@dataclass(frozen=True)
class _StaticSection:
    name: str
    duration_s: float | None = None

    def build(self, db_path: str) -> SummaryResult:
        payload = {
            "card": f"TraceML {self.name.replace('_', ' ').title()} Summary\n- Status: OK"
        }
        if self.duration_s is not None:
            payload["duration_s"] = self.duration_s
        return SummaryResult(
            section=self.name,
            payload=payload,
            text=payload["card"],
        )


@dataclass(frozen=True)
class _BrokenSection:
    name: str

    def build(self, db_path: str) -> SummaryResult:
        raise RuntimeError("section failed")


def _generator(*sections) -> FinalReportGenerator:
    return FinalReportGenerator(sections=sections)


def test_final_report_generator_preserves_summary_schema_and_order():
    payload = build_summary_payload(
        "fake.db",
        generator=_generator(
            _StaticSection("system"),
            _StaticSection("process", duration_s=12.5),
            _StaticSection("step_time", duration_s=10.0),
            _StaticSection("step_memory"),
        ),
    )

    assert payload["schema_version"] == 1.2
    assert payload["duration_s"] == 10.0
    assert list(payload.keys()) == [
        "schema_version",
        "generated_at",
        "duration_s",
        "system",
        "process",
        "step_time",
        "step_memory",
        "text",
    ]
    assert "TraceML Run Summary | duration 10.0s" in payload["text"]
    assert "System" in payload["text"]
    assert "Step Memory" in payload["text"]


def test_final_report_generator_fails_open_for_one_section():
    payload = build_summary_payload(
        "fake.db",
        generator=_generator(
            _StaticSection("system"),
            _BrokenSection("process"),
            _StaticSection("step_time"),
            _StaticSection("step_memory"),
        ),
    )

    assert payload["process"]["status"] == "NO DATA"
    assert payload["process"]["error"] == "Section summary unavailable."
    assert "Process" in payload["text"]


def test_reporting_final_is_the_summary_orchestration_owner():
    import traceml.reporting.final as reporting_final

    assert reporting_final.generate_summary is not None
    assert reporting_final.build_summary_payload is build_summary_payload
