from nicegui import ui
from traceml.renderers.display.nicegui_sections.system_section import (
    build_system_section,
    update_system_section,
)
from traceml.renderers.display.nicegui_sections.process_section import (
    build_process_section,
    update_process_section,
)

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

        with ui.row().classes("m-2 w-full gap-4 flex-nowrap items-start"):

            # System (left column)
            with ui.column().classes("w-1/2"):
                cls.cards["system_section"] = build_system_section()
                cls.update_funcs["system_section"] = update_system_section

            # Process (right column)
            with ui.column().classes("w-1/2"):
                cls.cards["process_section"] = build_process_section()
                cls.update_funcs["process_section"] = update_process_section

        # background update loop
        ui.timer(0.2, cls._ui_update_loop)
        cls._ui_ready = True