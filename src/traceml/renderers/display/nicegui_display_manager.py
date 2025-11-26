from nicegui import ui
import threading

class NiceGUIDisplayManager:
    _layout_content_fns = {}
    _ui_started = False

    @classmethod
    def start_display(cls):
        if cls._ui_started:
            return
        cls._ui_started = True

        cls.cards = {}

        # Run NiceGUI server in a background thread
        threading.Thread(target=lambda: ui.run(
            port=8765, reload=False, title="TraceML Dashboard", show=True
        ), daemon=True).start()

    @classmethod
    def register_layout_content(cls, layout_section: str, content_fn):
        if layout_section not in cls.cards:
            card = ui.card().classes('m-2 p-2')
            label = card.add(ui.label(f'{layout_section}: waiting...'))
            cls.cards[layout_section] = label

        cls._layout_content_fns[layout_section] = content_fn

    @classmethod
    def update_display(cls):
        for section, fn in cls._layout_content_fns.items():
            try:
                new_data = fn()
                cls.cards[section].text = str(new_data)
            except Exception as e:
                cls.cards[section].text = f"[Error] {e}"

    @classmethod
    def release_display(cls):
        pass  # nothing to clean up for NiceGUI
