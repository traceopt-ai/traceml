from nicegui import ui
import threading
import time

from traceml.renderers.display.nicegui_sections.system_section import (
    build_system_section,
    update_system_section,
)
from traceml.renderers.display.nicegui_sections.process_section import (
    build_process_section,
    update_process_section,
)


class NiceGUIDisplayManager:
    _layout_content_fns = {}
    _ui_started = False
    _ui_ready = False
    cards = {}
    update_funcs = {}
    latest_data = {}

    @classmethod
    def start_display(cls):
        if cls._ui_started:
            return
        cls._ui_started = True

        threading.Thread(target=cls._start_ui_server, daemon=True).start()

        while not cls._ui_ready:
            time.sleep(0.05)

    @classmethod
    def _start_ui_server(cls):

        @ui.page("/")
        def main_page():

            ui.label("TraceML Dashboard").classes("text-2xl m-2")
            with ui.column().classes("m-2"):
                # PREDEFINE SECTIONS
                cls.cards["system_section"] = build_system_section()
                cls.update_funcs["system_section"] = update_system_section

                cls.cards["process_section"] = build_process_section()
                cls.update_funcs["process_section"] = update_process_section
                # cls.cards['layer_summary'] = ui.label("layer_summary: waiting...")
                # cls.cards['activation_gradient'] = ui.label("activation_gradient: waiting...")
                # cls.cards['steptimer'] = ui.label("steptimer: waiting...")
            # UI TIMER — runs on UI thread every 0.2 sec
            ui.timer(0.2, cls._ui_update_loop)
            cls._ui_ready = True

        ui.run(port=8765, reload=False, show=True, title="TraceML Dashboard")

    @classmethod
    def _ui_update_loop(cls):
        if not cls._ui_ready:
            return
        for section, data in cls.latest_data.items():
            try:
                update_fn = cls.update_funcs.get(section)
                if update_fn:
                    update_fn(cls.cards[section], data)
            except Exception:
                for label in cls.cards[section].values():
                    label.text = "⚠️ Could not update"

    @classmethod
    def register_layout_content(cls, layout_section: str, content_fn):
        cls._layout_content_fns[layout_section] = content_fn

    @classmethod
    def update_display(cls):
        """Called from tracker thread → SAFE: stores data only."""
        if not cls._ui_ready:
            return

        for section, fn in cls._layout_content_fns.items():
            try:
                cls.latest_data[section] = fn()
            except Exception as e:
                cls.latest_data[section] = f"[Error] {e}"

    @classmethod
    def release_display(cls):
        pass
