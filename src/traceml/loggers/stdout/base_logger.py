from typing import Dict, Any

from .display_manager import StdoutDisplayManager


class BaseStdoutLogger:
    """
    Base class for specific stdout loggers. Each logger is responsible for
    providing data for a specific part of the shared display.
    """

    def __init__(self, name: str, layout_section_name: str):
        self.name = name
        self.layout_section_name = layout_section_name
        self._latest_data: Dict[str, Any] = {}

        # Ensure display is started and register this logger's content function
        StdoutDisplayManager.start_display()
        StdoutDisplayManager.register_layout_content(
            self.layout_section_name, self._get_panel_renderable
        )

    def _get_panel_renderable(self) -> Any:  # This will be implemented by subclasses
        """
        Abstract method: Subclasses must implement this to return a Rich Renderable
        (e.g., Panel, Table, Text) based on their `_latest_data`.
        """
        raise NotImplementedError(
            "Subclasses must implement _get_panel_renderable to provide content for the shared display."
        )

    def log(self, snapshots: Dict[str, Any]):
        """
        Receives snapshots from one or more samplers.
        Structure is always:
            { "SamplerClassName": { "ok": ..., "data": {...}, ... }, ... }
        """
        self._latest_env = snapshots
        self._latest_snapshot = snapshots

        StdoutDisplayManager.update_display()

    def log_summary(self, summary: Dict[str, Any]):
        """
        Abstract method: Subclasses must implement to log a final summary.
        This will typically be called after the main display is stopped.
        """
        raise NotImplementedError("Subclasses must implement log_summary method.")

    def shutdown(self):
        """
        Performs any specific shutdown for this logger.
        The main live display is stopped globally by the TrackerManager.
        """
        StdoutDisplayManager.release_display()
        print(
            f"[TraceML][{self.name}] Logger shutdown complete, {StdoutDisplayManager._active_logger_count} Loggers left."
        )
