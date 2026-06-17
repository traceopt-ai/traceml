"""Dashboard pages (PR2 revamp): brand chrome + Step-Time-first bento grid."""

from nicegui import app, ui

from traceml_ai.aggregator.display_drivers.layout import (
    LAYER_COMBINED_MEMORY_LAYOUT,
    LAYER_COMBINED_TIMER_LAYOUT,
    MODEL_COMBINED_LAYOUT,
    MODEL_DIAGNOSTICS_LAYOUT,
    MODEL_MEMORY_LAYOUT,
    PROCESS_LAYOUT,
    SYSTEM_LAYOUT,
)

from . import theme
from .layer_memory_table_section import (
    build_layer_memory_table_section,
    update_layer_memory_table_section,
)
from .layer_timer_table_section import (
    build_layer_timer_table_section,
    update_layer_timer_table_section,
)
from .model_combined_section import (
    build_model_combined_section,
    update_model_combined_section,
)
from .model_diagnostics_section import (
    build_model_diagnostics_section,
    update_model_diagnostics_section,
)
from .process_section import build_process_section, update_process_section
from .step_memory_section import (
    build_step_memory_section,
    update_step_memory_section,
)
from .system_section import (
    build_gpu_gauge_section,
    build_system_section,
    update_gpu_gauge_section,
    update_system_section,
)


def _run_context_text(payload) -> str:
    """Run config (world_size / GPU count / host) from the system payload ctx."""
    rollups = payload.get("rollups", {}) if isinstance(payload, dict) else {}
    ctx = rollups.get("ctx") if isinstance(rollups, dict) else None
    if not ctx:
        return ""
    ws = int(ctx.get("world_size") or 0)
    gc = int(ctx.get("gpu_count") or 0)
    host = str(ctx.get("hostname") or "").split(".")[0]
    parts = []
    if ws:
        parts.append(f"world_size {ws}")
    if gc:
        parts.append(f"{gc}-GPU node")
    if host:
        parts.append(host)
    return "  ·  ".join(parts)


def build_header(cls, show_layers: bool):
    """Brand run-context header: wordmark, run config, live-staleness chip."""
    with ui.element("div").classes("glass reveal").style("padding:15px 22px;"):
        with ui.row().classes("w-full items-center").style("gap:16px;"):
            with ui.row().style("gap:0; align-items:baseline;"):
                ui.label("Trace").classes("wm-trace")
                ui.label("ML").classes("wm-ml")
            ui.label("live training").classes("eyebrow")
            ctx = ui.label("").style(
                "font-family:var(--mono); font-size:11px; "
                "color:var(--muted); display:none;"
            )
            if show_layers:
                ui.link("layers", "/layers").style(
                    "font-family:var(--mono); font-size:12px; color:var(--orange-strong); "
                    "text-decoration:none; margin-left:4px;"
                )
            ui.element("div").style("flex:1;")
            staleness = (
                ui.label("").classes("staleband").style("display:none;")
            )
            cls.register_staleness_label(_StaleProxy(staleness))
            with ui.row().classes("items-center").style("gap:7px;"):
                ui.element("div").classes("livedot")
                ui.label("live").style(
                    "font-family:var(--mono); font-size:11px; color:#16a34a; font-weight:500;"
                )
    return ctx


class _StaleProxy:
    """Show the staleness chip only when there is text (hide when fresh)."""

    def __init__(self, label) -> None:
        self._label = label

    @property
    def text(self):
        return self._label.text

    @text.setter
    def text(self, value) -> None:
        self._label.text = value or ""
        self._label.style(f"display:{'inline-block' if value else 'none'};")


def _cell(flex: str):
    return ui.element("div").style(
        f"flex:{flex}; min-width:300px; display:flex; flex-direction:column;"
    )


def define_pages(cls):
    """Attach the NiceGUI pages with the revamped bento layout."""
    theme.register_static_fonts(app)
    deep_enabled = getattr(cls._settings, "profile", "run") == "deep"

    @ui.page("/")
    def main_page():
        ui.add_head_html(theme.head_html())
        with (
            ui.column()
            .classes("w-full")
            .style(
                "gap:16px; padding:22px 26px; max-width:1380px; margin:0 auto;"
            )
        ):
            header_ctx = build_header(cls, deep_enabled)

            # Row 1: hero (step-time ribbon + verdict) | GPU gauge
            with (
                ui.row()
                .classes("w-full items-stretch")
                .style("gap:16px; flex-wrap:wrap;")
            ):
                with _cell("2.4"):
                    cards = build_model_combined_section()
                    cls.subscribe_layout(
                        MODEL_COMBINED_LAYOUT,
                        cards,
                        update_model_combined_section,
                    )
                with _cell("1"):
                    gauge_cards = build_gpu_gauge_section()

            # Row 2: System | Process
            with (
                ui.row()
                .classes("w-full items-stretch")
                .style("gap:16px; flex-wrap:wrap;")
            ):
                with _cell("2"):
                    system_cards = build_system_section()
                with _cell("1.3"):
                    cards = build_process_section()
                    cls.subscribe_layout(
                        PROCESS_LAYOUT, cards, update_process_section
                    )

            # One SYSTEM_LAYOUT subscriber drives both the chart and the gauge
            # (two subscribers on one layout/client would evict each other).
            def _update_system(
                _c, d, _sc=system_cards, _gc=gauge_cards, _h=header_ctx
            ):
                update_system_section(_sc, d)
                update_gpu_gauge_section(_gc, d)
                txt = _run_context_text(d)
                if txt:
                    _h.text = txt
                    _h.style(
                        "font-family:var(--mono); font-size:11px; "
                        "color:var(--muted); display:inline-block;"
                    )

            cls.subscribe_layout(SYSTEM_LAYOUT, system_cards, _update_system)

            # Row 3: Step Memory | Diagnostics
            with (
                ui.row()
                .classes("w-full items-stretch")
                .style("gap:16px; flex-wrap:wrap;")
            ):
                with _cell("1.3"):
                    cards = build_step_memory_section()
                    cls.subscribe_layout(
                        MODEL_MEMORY_LAYOUT, cards, update_step_memory_section
                    )
                with _cell("1"):
                    cards = build_model_diagnostics_section()
                    cls.subscribe_layout(
                        MODEL_DIAGNOSTICS_LAYOUT,
                        cards,
                        update_model_diagnostics_section,
                    )

        cls.ensure_ui_timer(0.75)
        if not cls._ui_ready:
            cls._ui_ready = True

    if deep_enabled:

        @ui.page("/layers")
        def layer_page():
            ui.add_head_html(theme.head_html())
            with (
                ui.column()
                .classes("w-full")
                .style(
                    "gap:16px; padding:22px 26px; max-width:1380px; margin:0 auto;"
                )
            ):
                build_header(cls, True)
                with (
                    ui.row()
                    .classes("w-full items-stretch")
                    .style("gap:16px; flex-wrap:wrap;")
                ):
                    with _cell("1.2"):
                        cards = build_layer_memory_table_section()
                        cls.subscribe_layout(
                            LAYER_COMBINED_MEMORY_LAYOUT,
                            cards,
                            update_layer_memory_table_section,
                        )
                    with _cell("1"):
                        cards = build_layer_timer_table_section()
                        cls.subscribe_layout(
                            LAYER_COMBINED_TIMER_LAYOUT,
                            cards,
                            update_layer_timer_table_section,
                        )
            cls.ensure_ui_timer(1.0)
            if not cls._ui_ready:
                cls._ui_ready = True

    else:

        @ui.page("/layers")
        def layer_page_disabled():
            ui.navigate.to("/")
