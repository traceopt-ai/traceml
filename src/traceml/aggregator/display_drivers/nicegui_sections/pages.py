from nicegui import ui

from traceml.aggregator.display_drivers.layout import (
    LAYER_COMBINED_MEMORY_LAYOUT,
    LAYER_COMBINED_TIMER_LAYOUT,
    MODEL_COMBINED_LAYOUT,
    MODEL_DIAGNOSTICS_LAYOUT,
    MODEL_MEMORY_LAYOUT,
    PROCESS_LAYOUT,
    SYSTEM_LAYOUT,
)

from .layer_memory_table_section import (
    build_layer_memory_table_section,
    update_layer_memory_table_section,
)
from .layer_timer_table_section import (
    build_layer_timer_table_section,
    update_layer_timer_table_section,
)
from .model_combined_section import (
    build_model_combined_section,
    update_model_combined_section,
)
from .model_diagnostics_section import (
    build_model_diagnostics_section,
    update_model_diagnostics_section,
)
from .process_section import build_process_section, update_process_section
from .step_memory_section import (
    build_step_memory_section,
    update_step_memory_section,
)
from .system_section import build_system_section, update_system_section
from .ui_shell import PAGE_GAP_CLASS, VIEWPORT_STYLE


def build_top_tabs(active: str, show_layers: bool):
    """Shared top navigation tabs."""
    with ui.row().classes("w-full px-4 pt-1 pb-1 items-center"):
        ui.label("TraceML").classes("text-3xl font-extrabold mr-6").style(
            "color:#d47a00;"
        )
        with ui.tabs().classes("text-base") as tabs:
            overview = ui.tab("Overview")
            layers = ui.tab("Layer-wise") if show_layers else None

        tabs.value = (
            overview if active != "layers" or not show_layers else layers
        )
        overview.on("click", lambda: ui.navigate.to("/"))
        if show_layers and layers is not None:
            layers.on("click", lambda: ui.navigate.to("/layers"))


def define_pages(cls):
    """Attach the NiceGUI pages using a dense left-rail overview layout."""
    deep_enabled = getattr(cls._settings, "profile", "run") == "deep"

    @ui.page("/")
    def main_page():
        ui.add_head_html(
            """
            <style>
                body, .nicegui-content {
                    width: 100% !important;
                    max-width: 100% !important;
                    padding: 0 !important;
                    margin: 0 !important;
                }
                body {
                    background-color: #fff7f0 !important;
                    background-image: none !important;
                }
            </style>
            """
        )

        build_top_tabs(active="overview", show_layers=deep_enabled)

        with (
            ui.row()
            .classes(f"w-[99%] mx-2 {PAGE_GAP_CLASS} items-stretch")
            .style(VIEWPORT_STYLE)
        ):
            with (
                ui.column()
                .classes("h-full shrink-0")
                .style(
                    "width: 22%; min-width: 280px; max-width: 340px; overflow: hidden;"
                )
            ):
                cards = build_model_diagnostics_section()
                cls.subscribe_layout(
                    MODEL_DIAGNOSTICS_LAYOUT,
                    cards,
                    update_model_diagnostics_section,
                )

            with (
                ui.column()
                .classes(f"h-full flex-1 {PAGE_GAP_CLASS}")
                .style("min-width: 0; overflow: hidden;")
            ):
                with (
                    ui.row()
                    .classes(f"w-full {PAGE_GAP_CLASS} items-stretch no-wrap")
                    .style(
                        "height: 45%; min-height: 210px; flex-wrap: nowrap; overflow: hidden;"
                    )
                ):
                    with (
                        ui.column()
                        .classes("h-full flex-1")
                        .style("min-width: 0; overflow: hidden;")
                    ):
                        cards = build_system_section()
                        cls.subscribe_layout(
                            SYSTEM_LAYOUT, cards, update_system_section
                        )

                    with (
                        ui.column()
                        .classes("h-full flex-1")
                        .style("min-width: 0; overflow: hidden;")
                    ):
                        cards = build_process_section()
                        cls.subscribe_layout(
                            PROCESS_LAYOUT, cards, update_process_section
                        )

                with (
                    ui.row()
                    .classes(f"w-full {PAGE_GAP_CLASS} items-stretch no-wrap")
                    .style(
                        "height: 55%; min-height: 380px; flex-wrap: nowrap; overflow: hidden;"
                    )
                ):
                    with (
                        ui.column()
                        .classes("h-full shrink-0")
                        .style("width: 62%; min-width: 0; overflow: hidden;")
                    ):
                        cards = build_model_combined_section()
                        cls.subscribe_layout(
                            MODEL_COMBINED_LAYOUT,
                            cards,
                            update_model_combined_section,
                        )

                    with (
                        ui.column()
                        .classes("h-full flex-1")
                        .style("min-width: 0; overflow: hidden;")
                    ):
                        cards = build_step_memory_section()
                        cls.subscribe_layout(
                            MODEL_MEMORY_LAYOUT,
                            cards,
                            update_step_memory_section,
                        )

        cls.ensure_ui_timer(0.75)

        if not cls._ui_ready:
            cls._ui_ready = True

    if deep_enabled:

        @ui.page("/layers")
        def layer_page():
            build_top_tabs(active="layers", show_layers=True)

            with ui.row().classes("m-2 w-[99%] gap-2 flex-nowrap items-start"):
                with ui.column().classes("w-[54%]"):
                    cards = build_layer_memory_table_section()
                    cls.subscribe_layout(
                        LAYER_COMBINED_MEMORY_LAYOUT,
                        cards,
                        update_layer_memory_table_section,
                    )

                with ui.column().classes("w-[44%]"):
                    cards = build_layer_timer_table_section()
                    cls.subscribe_layout(
                        LAYER_COMBINED_TIMER_LAYOUT,
                        cards,
                        update_layer_timer_table_section,
                    )

            cls.ensure_ui_timer(1.0)

            if not cls._ui_ready:
                cls._ui_ready = True

    else:

        @ui.page("/layers")
        def layer_page_disabled():
            ui.navigate.to("/")
