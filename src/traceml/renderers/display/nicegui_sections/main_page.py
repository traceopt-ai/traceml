from nicegui import ui

from traceml.renderers.display.layout import (
    SYSTEM_LAYOUT,
    PROCESS_LAYOUT,
    LAYER_COMBINED_MEMORY_LAYOUT,
    ACTIVATION_GRADIENT_LAYOUT,
    STEPTIMER_LAYOUT,
    LAYER_COMBINED_TIMER_LAYOUT
)

from .system_section import (
    build_system_section,
    update_system_section,
)
from .process_section import (
    build_process_section,
    update_process_section,
)
from .layer_memory_table_section import (
    build_layer_memory_table_section,
    update_layer_memory_table_section
)
from .layer_timer_table_section import (
    build_layer_timer_table_section,
    update_layer_timer_table_section
)
from .steptiming_section import (
    build_step_timing_table_section,
    update_step_timing_table_section
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
                background-color: #fff7f0 !important;
                background-image: none !important;
            }
        </style>
        """)

        # ----- PAGE LAYOUT -----
        ui.label("TraceML Dashboard") \
            .classes("text-4xl font-extrabold m-4 w-full text-left") \
            .style("color:#d47a00;")

        with ui.row().classes("m-2 w-[99%] gap-4 flex-wrap items-center"):

            # System (left column)
            with ui.column().classes("w-[42%]"):
                cls.cards[SYSTEM_LAYOUT] = build_system_section()
                cls.update_funcs[SYSTEM_LAYOUT] = update_system_section

            # Process (middle column)
            with ui.column().classes("w-[30%]"):
                cls.cards[PROCESS_LAYOUT] = build_process_section()
                cls.update_funcs[PROCESS_LAYOUT] = update_process_section

            with ui.column().classes("w-[26]"):
                cls.cards[STEPTIMER_LAYOUT] = build_step_timing_table_section()
                cls.update_funcs[STEPTIMER_LAYOUT] = update_step_timing_table_section

        with ui.row().classes("m-2 w-[99%] gap-4 flex-nowrap items-center"):

            with ui.column().classes("w-[54%]"):
                cls.cards[LAYER_COMBINED_MEMORY_LAYOUT] = build_layer_memory_table_section()
                cls.update_funcs[LAYER_COMBINED_MEMORY_LAYOUT] = update_layer_memory_table_section

            with ui.column().classes("w-[44%]"):
                cls.cards[LAYER_COMBINED_TIMER_LAYOUT] = build_layer_timer_table_section()
                cls.update_funcs[LAYER_COMBINED_TIMER_LAYOUT] = update_layer_timer_table_section


        for l in [
            ACTIVATION_GRADIENT_LAYOUT
        ]:
            cls.cards[l] = build_fake_section()
            cls.update_funcs[l] = update_fake_section

        # background update loop
        ui.timer(0.75, cls._ui_update_loop)
        cls._ui_ready = True