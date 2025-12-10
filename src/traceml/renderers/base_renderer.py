from typing import Dict, Any


class BaseRenderer:
    """
    Base class for specific stdout loggers. Each logger is responsible for
    providing data for a specific part of the shared display.
    """

    def __init__(self, name: str, layout_section_name: str):
        self.name = name
        self.layout_section_name = layout_section_name
        self._latest_data: Dict[str, Any] = {}

    def get_panel_renderable(self) -> Any:  # This will be implemented by subclasses
        """
        Abstract method: Subclasses must implement this to return a Rich Renderable
        (e.g., Panel, Table, Text) based on their `_latest_data`.
        """
        raise NotImplementedError(
            "Subclasses must implement _get_panel_renderable to provide content for the shared display."
        )

    def get_notebook_renderable(self) -> Any:
        """
        Subclasses implement this to return an HTML representation
        (IPython.display.HTML) based on `get_data()`.
        Used in Jupyter/Notebook display.
        """
        raise NotImplementedError("Subclasses must implement get_notebook_renderable()")

    def log_summary(self):
        """
        Abstract method: Subclasses must implement to log a final summary.
        This will typically be called after the main display is stopped.
        """
        raise NotImplementedError("Subclasses must implement log_summary method.")
