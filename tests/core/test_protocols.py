from traceml.core import (
    Formatter,
    RenderContext,
    Startable,
    SummaryResult,
)


class ExampleStartable:
    def __init__(self) -> None:
        self.started = False

    def start(self) -> None:
        self.started = True


class ExampleFormatter:
    name = "example"

    def format(self, payload: dict[str, int]) -> str:
        return str(payload["value"])


def test_lifecycle_protocol_accepts_structural_implementation() -> None:
    component = ExampleStartable()

    assert isinstance(component, Startable)


def test_render_context_defaults_are_isolated() -> None:
    first = RenderContext()
    second = RenderContext()

    first.options["window"] = 100

    assert second.options == {}


def test_summary_result_has_json_payload_and_text_card() -> None:
    result = SummaryResult(
        section="system",
        payload={"status": "healthy"},
        text="system healthy",
    )

    assert result.section == "system"
    assert result.payload["status"] == "healthy"
    assert result.text == "system healthy"


def test_formatter_protocol_describes_format_method() -> None:
    formatter: Formatter[dict[str, int], str] = ExampleFormatter()

    assert formatter.format({"value": 7}) == "7"
