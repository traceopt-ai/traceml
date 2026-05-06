from traceml.renderers.base_renderer import (
    BaseRenderer,
    CLIRenderer,
    DashboardRenderer,
    RendererMetadata,
)


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
