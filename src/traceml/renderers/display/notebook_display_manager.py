from typing import Dict, Callable, Any
from IPython.display import display
import ipywidgets as widgets

from traceml.loggers.error_log import get_error_logger


class NotebookDisplayManager:
    """
    Notebook display manager using ipywidgets.
    """

    _layout_content_fns: Dict[str, Callable[[], Any]] = {}
    _widgets: Dict[str, widgets.HTML] = {}
    _container: widgets.VBox = None
    _active_logger_count: int = 0

    logger = get_error_logger("NotebookDisplayManager")

    @classmethod
    def start_display(cls):
        """Initialize a VBox container and show it once."""
        if cls._active_logger_count == 0:
            cls._container = widgets.VBox([])
            display(cls._container)
        cls._active_logger_count += 1

    @classmethod
    def stop_display(cls):
        """Stop display and clear state."""
        cls._layout_content_fns.clear()
        cls._widgets.clear()
        cls._container = None
        cls._active_logger_count = 0

    @classmethod
    def release_display(cls):
        """Decrement counter and stop if no loggers remain."""
        cls._active_logger_count = max(cls._active_logger_count - 1, 0)
        if cls._active_logger_count == 0:
            cls.stop_display()

    @classmethod
    def register_layout_content(
        cls, layout_section: str, content_fn: Callable[[], Any]
    ):
        """Register a logger section and create its widget if needed."""
        cls._layout_content_fns[layout_section] = content_fn
        if layout_section not in cls._widgets:
            html_widget = widgets.HTML(
                value=f"<div style='color:gray;'>Waiting for {layout_section}...</div>"
            )
            cls._widgets[layout_section] = html_widget
            if cls._container:
                cls._container.children = list(cls._widgets.values())

    @classmethod
    def update_display(cls):
        """Update widgets in-place."""
        for section_name, content_fn in cls._layout_content_fns.items():
            try:
                html_obj = content_fn()
                if html_obj is not None:
                    cls._widgets[section_name].value = html_obj.data
            except Exception as e:
                cls._widgets[section_name].value = (
                    f"<div style='color:red;'>Error rendering {section_name}: {e}</div>"
                )
                cls.logger.error(
                    f"[TraceML] Error rendering notebook section {section_name}: {e}"
                )
