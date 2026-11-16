"""Dashboard pages: context strip, then the sections the profile owns."""

from nicegui import app, ui

from traceml_ai.aggregator.display_drivers.layout import (
    CONTEXT_LAYOUT,
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
    should_show_step_memory_card,
    update_step_memory_section,
)
from .system_section import build_system_section, update_system_section


def _cell(flex: str):
    return ui.element("div").style(
        f"flex:{flex}; min-width:300px; display:flex; flex-direction:column;"
    )


def _remember_system_gpu_availability(state, data) -> None:
    """Record GPU status only after the system payload has evidence."""
    if getattr(data, "gpu_available", False):
        state["gpu_available"] = True
    elif getattr(data, "window_len", 0):
        state["gpu_available"] = False


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
        dashboard_state = {
            "gpu_available": None,
            "step_memory_has_data": False,
            "step_memory_panel": None,
        }

        def _sync_step_memory_visibility():
            panel = dashboard_state["step_memory_panel"]
            if panel is None:
                return
            visible = should_show_step_memory_card(
                gpu_available=dashboard_state["gpu_available"],
                has_memory_data=dashboard_state["step_memory_has_data"],
            )
            panel["container"].set_visibility(visible)

        def _update_system(cards, data):
            update_system_section(cards, data)
            # ``gpu_available=False`` is only conclusive after a sample; the
            # empty pre-sample payload uses the same default value.
            _remember_system_gpu_availability(dashboard_state, data)
            _sync_step_memory_visibility()

        def _update_step_memory(cards, data):
            update_step_memory_section(cards, data)
            metrics = getattr(data, "metrics", None)
            if metrics is None and isinstance(data, dict):
                metrics = data.get("metrics")
            dashboard_state["step_memory_has_data"] = bool(metrics)
            _sync_step_memory_visibility()

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
            cls.subscribe_layout(CONTEXT_LAYOUT, strip, update_context_section)

            hero_cards = None
            if "model_combined" in shown:
                # Row 1: hero (step-time ribbon + verdict). A watch session
                # has no step loop, so no hero: the resource panes lead.
                with (
                    ui.row()
                    .classes("w-full items-stretch")
                    .style("gap:16px; flex-wrap:wrap;")
                ):
                    with _cell("1"):
                        hero_cards = build_model_combined_section()
                        cls.subscribe_layout(
                            MODEL_COMBINED_LAYOUT,
                            hero_cards,
                            update_model_combined_section,
                        )

            # System | Process: the evidence pair (the node, its ranks),
            # side by side at exactly half the width each. Heights are not
            # stretched to match (items-start): the two blocks share one
            # skeleton (a tiles row, fixed-height chart slots, one
            # disclosure), so they are equal by construction while their
            # disclosures are closed, and an opened disclosure grows only
            # its own card. Process adopts the skeleton in its own PR; until
            # then it is today's card. The System block carries the GPU
            # headline itself (util tile + per-GPU rows), so there is no
            # gauge.
            with (
                ui.row()
                .classes("w-full items-start")
                .style("gap:16px; flex-wrap:wrap;")
            ):
                with _cell("1 1 0"):
                    system_cards = build_system_section()
                with _cell("1 1 0"):
                    cards = build_process_section()
                    cls.subscribe_layout(
                        PROCESS_LAYOUT, cards, update_process_section
                    )

            cls.subscribe_layout(SYSTEM_LAYOUT, system_cards, _update_system)

            # Row 3: Step Memory | Diagnostics (training runs only)
            if "step_memory" in shown or "model_diagnostics" in shown:
                with (
                    ui.row()
                    .classes("w-full items-stretch")
                    .style("gap:16px; flex-wrap:wrap;")
                ):
                    if "step_memory" in shown:
                        step_memory_container = _cell("1.3")
                        with step_memory_container:
                            cards = build_step_memory_section()
                        cards["container"] = step_memory_container
                        dashboard_state["step_memory_panel"] = cards
                        cls.subscribe_layout(
                            MODEL_MEMORY_LAYOUT,
                            cards,
                            _update_step_memory,
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
