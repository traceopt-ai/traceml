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
                cls.cards["system_section"] = build_system_section()
                cls.update_funcs["system_section"] = update_system_section

            # Process (right column)
            with ui.column().classes("w-[50%]"):
                cls.cards["process_section"] = build_process_section()
                cls.update_funcs["process_section"] = update_process_section

        with ui.row().classes("m-2 w-[90%] gap-4 flex-nowrap items-center"):
            cls.cards["layer_combined_summary_section"] = build_layer_table_section()
            cls.update_funcs["layer_combined_summary_section"] = update_layer_table_section

        # background update loop
        ui.timer(0.75, cls._ui_update_loop)
        cls._ui_ready = True