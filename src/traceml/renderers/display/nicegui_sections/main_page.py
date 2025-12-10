from nicegui import ui
from traceml.renderers.display.nicegui_sections.system_section import (
    build_system_section,
    update_system_section,
)
from traceml.renderers.display.nicegui_sections.process_section import (
    build_process_section,
    update_process_section,
)
from traceml.renderers.display.nicegui_sections.layer_table_section import (
    build_layer_table_section,
    update_layer_table_section,
)

from traceml.renderers.display.layout import (
    SYSTEM_LAYOUT_NAME,
    PROCESS_LAYOUT_NAME,
    LAYER_COMBINED_LAYOUT_NAME,
    ACTIVATION_GRADIENT_LAYOUT_NAME,
    STEPTIMER_LAYOUT_NAME,
    STDOUT_STDERR_LAYOUT_NAME
)

from .helper import build_fake_section, update_fake_section
import pandas

def define_main_page(cls):
    """Attach the NiceGUI main page to the UI server."""

    @ui.page("/")
    def main_page():

        # ----- GLOBAL PAGE STYLES -----
        ui.add_head_html("""
        <style>
            body, .nicegui-content {
                width: 100% !important;
                max-width: 100% !important;
                padding: 0 !important;
                margin: 0 !important;
            }
            body {
                background-color: #e5e5e5 !important;
                background-image: none !important;
            }
        </style>
        """)

        # ----- PAGE LAYOUT -----
        ui.label("TraceML").classes("text-2xl m-2")

        with ui.row().classes("m-2 w-[90%] gap-4 flex-nowrap items-center"):

            # System (left column)
            with ui.column().classes("w-[50%]"):
                cls.cards[SYSTEM_LAYOUT_NAME] = build_system_section()
                cls.update_funcs[SYSTEM_LAYOUT_NAME] = update_system_section

            # Process (right column)
            with ui.column().classes("w-[50%]"):
                cls.cards[PROCESS_LAYOUT_NAME] = build_process_section()
                cls.update_funcs[PROCESS_LAYOUT_NAME] = update_process_section

        with ui.row().classes("m-2 w-[90%] gap-4 flex-nowrap items-center"):
            cls.cards[LAYER_COMBINED_LAYOUT_NAME] = build_layer_table_section()
            cls.update_funcs[LAYER_COMBINED_LAYOUT_NAME] = update_layer_table_section

        for l in [
            ACTIVATION_GRADIENT_LAYOUT_NAME,
            STEPTIMER_LAYOUT_NAME,
            STDOUT_STDERR_LAYOUT_NAME
        ]:
            cls.cards[l] = build_fake_section()
            cls.update_funcs[l] = update_fake_section

        # background update loop
        ui.timer(0.75, cls._ui_update_loop)
        cls._ui_ready = True