from typing import Dict, Callable, Any
from IPython.display import display
import ipywidgets as widgets

from traceml.loggers.error_log import get_error_logger
from traceml.renderers.display.page_layout import TRACE_ML_PAGE


class NotebookDisplayManager:
    """
    Notebook display manager using ipywidgets.

    - Single page
    - Shared layout via TRACE_ML_PAGE
    - One render pass per update
    """

    _content_fns: Dict[str, Callable[[], Any]] = {}
    _page: widgets.HTML | None = None
    _active_logger_count: int = 0

    logger = get_error_logger("NotebookDisplayManager")

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    @classmethod
    def start_display(cls):
        """Create and display the page once."""
        if cls._active_logger_count == 0:
            cls._page = widgets.HTML(
                value="<div style='color:gray;'>Initializing TraceML…</div>"
            )
            display(cls._page)
        cls._active_logger_count += 1

    @classmethod
    def stop_display(cls):
        """Clear all state."""
        cls._content_fns.clear()
        cls._page = None
        cls._active_logger_count = 0

    @classmethod
    def release_display(cls):
        """Decrement refcount and stop if unused."""
        cls._active_logger_count = max(cls._active_logger_count - 1, 0)
        if cls._active_logger_count == 0:
            cls.stop_display()

    # --------------------------------------------------
    # Registration
    # --------------------------------------------------

    @classmethod
    def register_layout_content(
        cls, layout_section: str, content_fn: Callable[[], Any]
    ):
        """
        Register a section render function.

        content_fn MUST return IPython.display.HTML
        """
        cls._content_fns[layout_section] = content_fn

    # --------------------------------------------------
    # Rendering
    # --------------------------------------------------

    @classmethod
    def update_display(cls):
        """Re-render the full page from the latest section content."""
        if cls._page is None:
            return

        sections_html: Dict[str, str] = {}

        for name, fn in cls._content_fns.items():
            try:
                html_obj = fn()
                if html_obj is not None:
                    sections_html[name] = html_obj.data
            except Exception as e:
                sections_html[name] = (
                    f"<div style='color:red;'>"
                    f"Error rendering {name}: {e}"
                    f"</div>"
                )
                cls.logger.error(
                    f"[TraceML] Error rendering notebook section {name}: {e}"
                )

        cls._page.value = cls._render_page(sections_html)

    # --------------------------------------------------
    # Layout (single source of truth)
    # --------------------------------------------------

    @classmethod
    def _render_page(cls, sections: Dict[str, str]) -> str:
        """Render the full notebook page using TRACE_ML_PAGE."""

        def section_html(name: str) -> str:
            return sections.get(
                name,
                f"<div style='color:gray;'>Waiting for {name}…</div>",
            )

        rows_html = []

        for row in TRACE_ML_PAGE:
            cols_html = []
            for section in row:
                cols_html.append(
                    f"""
                    <div style="flex:1; min-width:0;">
                        {section_html(section)}
                    </div>
                    """
                )

            rows_html.append(
                f"""
                <div style="
                    display:flex;
                    gap:16px;
                    align-items:stretch;
                ">
                    {''.join(cols_html)}
                </div>
                """
            )

        return f"""
        <div style="
            display:flex;
            flex-direction:column;
            gap:16px;
            font-family:Arial, sans-serif;
        ">
            {''.join(rows_html)}
        </div>
        """



#
# class NotebookDisplayManager:
#     """
#     Notebook display manager using ipywidgets.
#     """
#
#     _layout_content_fns: Dict[str, Callable[[], Any]] = {}
#     _widgets: Dict[str, widgets.HTML] = {}
#     _container: widgets.VBox = None
#     _active_logger_count: int = 0
#
#     logger = get_error_logger("NotebookDisplayManager")
#
#     @classmethod
#     def start_display(cls):
#         """Initialize a VBox container and show it once."""
#         if cls._active_logger_count == 0:
#             cls._container = widgets.VBox([])
#             display(cls._container)
#         cls._active_logger_count += 1
#
#     @classmethod
#     def stop_display(cls):
#         """Stop display and clear state."""
#         cls._layout_content_fns.clear()
#         cls._widgets.clear()
#         cls._container = None
#         cls._active_logger_count = 0
#
#     @classmethod
#     def release_display(cls):
#         """Decrement counter and stop if no loggers remain."""
#         cls._active_logger_count = max(cls._active_logger_count - 1, 0)
#         if cls._active_logger_count == 0:
#             cls.stop_display()
#
#     @classmethod
#     def register_layout_content(
#         cls, layout_section: str, content_fn: Callable[[], Any]
#     ):
#         """Register a logger section and create its widget if needed."""
#         cls._layout_content_fns[layout_section] = content_fn
#         if layout_section not in cls._widgets:
#             html_widget = widgets.HTML(
#                 value=f"<div style='color:gray;'>Waiting for {layout_section}...</div>"
#             )
#             cls._widgets[layout_section] = html_widget
#             if cls._container:
#                 cls._container.children = list(cls._widgets.values())
#
#     @classmethod
#     def update_display(cls):
#         """Update widgets in-place."""
#         for section_name, content_fn in cls._layout_content_fns.items():
#             try:
#                 html_obj = content_fn()
#                 if html_obj is not None:
#                     cls._widgets[section_name].value = html_obj.data
#             except Exception as e:
#                 cls._widgets[section_name].value = (
#                     f"<div style='color:red;'>Error rendering {section_name}: {e}</div>"
#                 )
#                 cls.logger.error(
#                     f"[TraceML] Error rendering notebook section {section_name}: {e}"
#                 )
