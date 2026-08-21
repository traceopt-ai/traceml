"""Dashboard pages: context strip, then the sections the profile owns."""

from nicegui import app, ui

from traceml_ai.aggregator.display_drivers.layout import (
    MODEL_COMBINED_LAYOUT,
    MODEL_DIAGNOSTICS_LAYOUT,
    MODEL_MEMORY_LAYOUT,
    PROCESS_LAYOUT,
    SYSTEM_LAYOUT,
)

from . import theme
from .context_section import (
    build_context_section,
    resolve_run_identity,
    sections_for_profile,
    update_context_section,
)
from .model_combined_section import (
    build_model_combined_section,
    update_model_combined_section,
    update_step_verdict,
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


def _cell(flex: str):
    return ui.element("div").style(
        f"flex:{flex}; min-width:300px; display:flex; flex-direction:column;"
    )


def define_pages(cls):
    """Attach the NiceGUI pages: context strip, then the profile's sections."""
    theme.register_static_fonts(app)
    profile = str(getattr(cls._settings, "profile", "run") or "run")
    deep_enabled = profile == "deep"
    shown = sections_for_profile(profile)
    identity = resolve_run_identity(cls._settings)

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
            strip = build_context_section(
                identity,
                cls.register_staleness_label,
                show_layers=deep_enabled,
                sampler_interval_s=getattr(
                    cls._settings, "sampler_interval_sec", None
                ),
            )

            hero_cards = None
            if "model_combined" in shown:
                # Row 1: hero (step-time ribbon + verdict) | GPU gauge
                with (
                    ui.row()
                    .classes("w-full items-stretch")
                    .style("gap:16px; flex-wrap:wrap;")
                ):
                    with _cell("2.4"):
                        hero_cards = build_model_combined_section()
                        cls.subscribe_layout(
                            MODEL_COMBINED_LAYOUT,
                            hero_cards,
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
            else:
                # A watch session has no step loop, so no hero: the
                # resource panes take the page. Row 1: System | GPU gauge,
                # Row 2: Process.
                with (
                    ui.row()
                    .classes("w-full items-stretch")
                    .style("gap:16px; flex-wrap:wrap;")
                ):
                    with _cell("2"):
                        system_cards = build_system_section()
                    with _cell("1"):
                        gauge_cards = build_gpu_gauge_section()
                with (
                    ui.row()
                    .classes("w-full items-stretch")
                    .style("gap:16px; flex-wrap:wrap;")
                ):
                    with _cell("1"):
                        cards = build_process_section()
                        cls.subscribe_layout(
                            PROCESS_LAYOUT, cards, update_process_section
                        )

            # One SYSTEM_LAYOUT subscriber drives the chart, the gauge and
            # the context strip (two subscribers on one layout/client would
            # evict each other).
            def _update_system(
                _c, d, _sc=system_cards, _gc=gauge_cards, _st=strip
            ):
                update_system_section(_sc, d)
                update_gpu_gauge_section(_gc, d)
                update_context_section(_st, d)

            cls.subscribe_layout(SYSTEM_LAYOUT, system_cards, _update_system)

            # Row 3: Step Memory | Diagnostics (training runs only)
            if "step_memory" in shown or "model_diagnostics" in shown:
                with (
                    ui.row()
                    .classes("w-full items-stretch")
                    .style("gap:16px; flex-wrap:wrap;")
                ):
                    if "step_memory" in shown:
                        with _cell("1.3"):
                            cards = build_step_memory_section()
                            cls.subscribe_layout(
                                MODEL_MEMORY_LAYOUT,
                                cards,
                                update_step_memory_section,
                            )
                    if "model_diagnostics" in shown:
                        with _cell("1"):
                            diag_cards = build_model_diagnostics_section()

                            # One MODEL_DIAGNOSTICS_LAYOUT subscriber drives
                            # BOTH the rail and the hero verdict, so the hero
                            # shows the engine's step-time status verbatim
                            # (single source of truth). Two subscribers on
                            # one layout/client would evict each other.
                            def _update_diag(
                                _c, d, _dc=diag_cards, _hc=hero_cards
                            ):
                                update_model_diagnostics_section(_dc, d)
                                if _hc is not None:
                                    update_step_verdict(_hc, d)

                            cls.subscribe_layout(
                                MODEL_DIAGNOSTICS_LAYOUT,
                                diag_cards,
                                _update_diag,
                            )

        cls.ensure_ui_timer(0.75)
        if not cls._ui_ready:
            cls._ui_ready = True
