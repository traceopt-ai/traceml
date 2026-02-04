from nicegui import ui
import threading
import time
from traceml.renderers.display.nicegui_sections.pages import define_pages


class NiceGUIDisplayManager:
    _layout_content_fns = {}
    _ui_started = False
    _ui_ready = False
    _latest_data_lock = threading.Lock()
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
        define_pages(cls)
        ui.run(port=8765, reload=False, show=True, title="TraceML Dashboard")

    @classmethod
    def _ui_update_loop(cls):
        if not cls._ui_ready:
            return
        with cls._latest_data_lock:
            for layout, data in cls.latest_data.items():
                try:
                    update_fn = cls.update_funcs.get(layout)
                    if update_fn:
                        update_fn(cls.cards[layout], data)
                except Exception:
                    for label in cls.cards[layout].values():
                        label.text = "⚠️ Could not update"

    @classmethod
    def register_layout_content(cls, layout_section: str, content_fn):
        cls._layout_content_fns[layout_section] = content_fn

    @classmethod
    def update_display(cls):
        """Called from tracker thread → SAFE: stores data only."""
        if not cls._ui_ready:
            return
        with cls._latest_data_lock:
            for layout, fn in cls._layout_content_fns.items():
                try:
                    cls.latest_data[layout] = fn()
                except Exception as e:
                    cls.latest_data[layout] = f"[Error] {e}"

    @classmethod
    def release_display(cls):
        """
        Gracefully shut down the NiceGUI display server.

        This must be called from the TraceML runtime shutdown path.
        It is safe to call multiple times.
        """
        if not cls._ui_started:
            return

        cls._ui_started = False
        cls._ui_ready = False

        # Stop UI updates
        with cls._latest_data_lock:
            cls.latest_data.clear()
            cls.cards.clear()
            cls.update_funcs.clear()
            cls._layout_content_fns.clear()
