import builtins
from unittest.mock import Mock

import pytest

from traceml_ai.aggregator.display_drivers.cli import CLIDisplayDriver
from traceml_ai.aggregator.display_drivers.layout import STDOUT_STDERR_LAYOUT
from traceml_ai.aggregator.sqlite_writers import stdout_stderr
from traceml_ai.aggregator.trace_aggregator import _resolve_display_driver
from traceml_ai.renderers.base_renderer import (
    BaseRenderer,
    CLIRenderer,
    DashboardRenderer,
    RendererMetadata,
)
from traceml_ai.renderers.stdout_stderr.common import StdoutStderrDB
from traceml_ai.runtime.settings import TraceMLSettings


class MetadataOnlyRenderer(BaseRenderer):
    pass


class ExampleCLIRenderer(BaseRenderer):
    def __init__(self) -> None:
        super().__init__(name="cli", layout_section_name="cli_section")

    def get_panel_renderable(self) -> str:
        return "panel"


class ExampleDashboardRenderer(BaseRenderer):
    def __init__(self) -> None:
        super().__init__(
            name="dashboard",
            layout_section_name="dashboard_section",
        )

    def get_dashboard_renderable(self) -> dict[str, str]:
        return {"payload": "dashboard"}


class ExampleDualRenderer(ExampleCLIRenderer):
    def get_dashboard_renderable(self) -> dict[str, str]:
        return {"payload": "dual"}


def test_base_renderer_only_owns_shared_metadata() -> None:
    renderer = MetadataOnlyRenderer(
        name="metadata",
        layout_section_name="section",
    )

    assert renderer.name == "metadata"
    assert renderer.layout_section_name == "section"
    assert renderer._latest_data == {}
    assert isinstance(renderer, RendererMetadata)
    assert not hasattr(renderer, "get_notebook_renderable")


def test_cli_renderer_contract_is_separate_from_dashboard() -> None:
    renderer = ExampleCLIRenderer()

    assert isinstance(renderer, CLIRenderer)
    assert not isinstance(renderer, DashboardRenderer)
    assert renderer.get_panel_renderable() == "panel"


def test_dashboard_renderer_contract_is_separate_from_cli() -> None:
    renderer = ExampleDashboardRenderer()

    assert isinstance(renderer, DashboardRenderer)
    assert not isinstance(renderer, CLIRenderer)
    assert renderer.get_dashboard_renderable() == {"payload": "dashboard"}


def test_renderer_can_support_both_cli_and_dashboard_contracts() -> None:
    renderer = ExampleDualRenderer()

    assert isinstance(renderer, CLIRenderer)
    assert isinstance(renderer, DashboardRenderer)
    assert renderer.get_panel_renderable() == "panel"
    assert renderer.get_dashboard_renderable() == {"payload": "dual"}


@pytest.mark.parametrize("profile", ["watch", "run"])
def test_cli_layout_has_no_legacy_stdout_panel(profile) -> None:
    driver = CLIDisplayDriver(
        logger=Mock(),
        settings=TraceMLSettings(profile=profile, mode="cli"),
    )

    driver._create_initial_layout()

    assert not driver._has_section(STDOUT_STDERR_LAYOUT)
    assert all(
        renderer.layout_section_name != STDOUT_STDERR_LAYOUT
        for renderer in driver._renderers
    )


def test_legacy_stdout_history_remains_readable(tmp_path) -> None:
    db_path = tmp_path / "legacy.db"
    reader = StdoutStderrDB(str(db_path))
    with reader.connect() as conn:
        stdout_stderr.init_schema(conn)
        conn.execute(
            """
            INSERT INTO stdout_stderr_samples(
                recv_ts_ns, rank, sample_ts_s, line
            ) VALUES (1, 2, 3.0, 'legacy output');
            """
        )
        lines = reader.fetch_latest_lines(conn, rank=2)

    assert [line.line for line in lines] == ["legacy output"]


def test_dashboard_driver_missing_dependency_has_install_hint(
    monkeypatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "traceml_ai.aggregator.display_drivers.nicegui":
            raise ModuleNotFoundError(
                "No module named 'nicegui'",
                name="nicegui",
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="pip install -U traceml-ai"):
        _resolve_display_driver("dashboard")
