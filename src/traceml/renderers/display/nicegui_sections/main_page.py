from nicegui import ui

from traceml.renderers.display.layout import (
    LAYER_COMBINED_MEMORY_LAYOUT,
    LAYER_COMBINED_TIMER_LAYOUT,
    MODEL_COMBINED_LAYOUT,
    PROCESS_LAYOUT,
    STEPTIMER_LAYOUT,
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
from .process_section import build_process_section, update_process_section
from .steptiming_section import (
    build_step_timing_table_section,
    update_step_timing_table_section,
)
from .system_section import build_system_section, update_system_section


def fake_build():
    pass


def fake_update():
    pass


def build_top_tabs(active: str):
    """
    Shared top navigation tabs.
    `active` ∈ {"overview", "layers"}
    """
    with ui.row().classes("w-full px-4 pt-2"):
        with ui.tabs().classes("text-lg") as tabs:
            overview = ui.tab("Overview")
            layers = ui.tab("Layer-wise")

        if active == "overview":
            tabs.value = overview
        else:
            tabs.value = layers

        overview.on("click", lambda: ui.navigate.to("/"))
        layers.on("click", lambda: ui.navigate.to("/layers"))


def define_main_page(cls):
    """Attach the NiceGUI main page to the UI server."""

    @ui.page("/")
    def main_page():

        # ----- GLOBAL PAGE STYLES -----
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
        # ----- PAGE LAYOUT -----
        ui.label("TraceML").classes(
            "text-4xl font-extrabold mt-3 mb-1 ml-4 w-full text-left"
        ).style("color:#d47a00;")

        build_top_tabs(active="overview")

        with ui.row().classes("mt-1 mx-2 w-[99%] gap-2 flex-nowrap items-center"):

            # System (left column)
            with ui.column().classes("w-[36%]"):
                cls.cards[SYSTEM_LAYOUT] = build_system_section()
                cls.update_funcs[SYSTEM_LAYOUT] = update_system_section

            # Process (middle column)
            with ui.column().classes("w-[30%]"):
                cls.cards[PROCESS_LAYOUT] = build_process_section()
                cls.update_funcs[PROCESS_LAYOUT] = update_process_section

            with ui.column().classes("w-[33]"):
                cls.cards[STEPTIMER_LAYOUT] = build_step_timing_table_section()
                cls.update_funcs[STEPTIMER_LAYOUT] = update_step_timing_table_section

        with ui.row().classes("m-2 w-[99%] gap-2 flex-nowrap items-center"):
            with ui.column().classes("w-[99%]"):
                cls.cards[MODEL_COMBINED_LAYOUT] = build_model_combined_section()
                cls.update_funcs[MODEL_COMBINED_LAYOUT] = update_model_combined_section

        # background update loop
        ui.timer(0.75, cls._ui_update_loop)
        cls._ui_ready = True

    @ui.page("/layers")
    def layer_page():
        ui.label("TraceML").classes(
            "text-4xl font-extrabold mt-3 mb-1 ml-4 w-full text-left"
        ).style("color:#d47a00;")

        build_top_tabs(active="layers")

        with ui.row().classes("m-2 w-[99%] gap-2 flex-nowrap items-start"):
            with ui.column().classes("w-[54%]"):
                cls.cards[LAYER_COMBINED_MEMORY_LAYOUT] = (
                    build_layer_memory_table_section()
                )
                cls.update_funcs[LAYER_COMBINED_MEMORY_LAYOUT] = (
                    update_layer_memory_table_section
                )

            with ui.column().classes("w-[44%]"):
                cls.cards[LAYER_COMBINED_TIMER_LAYOUT] = (
                    build_layer_timer_table_section()
                )
                cls.update_funcs[LAYER_COMBINED_TIMER_LAYOUT] = (
                    update_layer_timer_table_section
                )

        ui.timer(0.75, cls._ui_update_loop)
