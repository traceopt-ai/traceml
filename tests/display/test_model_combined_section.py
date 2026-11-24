from __future__ import annotations

import json
import re
import sqlite3

import pytest
from rich.console import Console

from tests.step_time.factories import (
    live_result_from_window,
    rank_average,
    window_from_rank_averages,
)

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui_sections import theme
from traceml_ai.aggregator.display_drivers.nicegui_sections.model_combined_section import (
    build_model_combined_section,
    update_model_combined_section,
)
from traceml_ai.diagnostics.model_diagnostics import (
    build_model_diagnostics_payload,
)
from traceml_ai.renderers.step_time.renderer import (
    StepTimeRenderer,
    format_cli_diagnosis,
)
from traceml_ai.reporting.sections.step_time import StepTimeSummarySection
from traceml_ai.step_time.model import (
    StepTimeLoadRequest,
    StepTimeMetric,
    StepTimeWindow,
)
from traceml_ai.step_time.pipeline import (
    LiveStepTimeFreshness,
    LiveStepTimeResult,
    LiveStepTimeSession,
)


class _FakeText:
    def __init__(self) -> None:
        self.text = ""
        self.styles: list[str] = []

    def style(self, value: str) -> "_FakeText":
        self.styles.append(value)
        return self


class _FakeHtml:
    def __init__(self) -> None:
        self.content = ""


def _metric(
    name: str,
    value: float,
    worst: float | None = None,
) -> StepTimeMetric:
    return StepTimeMetric(
        metric=name,
        series=None,
        median_total=value,
        worst_total=worst if worst is not None else value,
        worst_rank=0,
        skew_pct=0.0,
    )


_COMPLETE_METRIC_VALUES = {
    "input_wait": 10.0,
    "h2d": 10.0,
    "forward": 20.0,
    "backward": 30.0,
    "optimizer_step": 20.0,
    "residual_proxy": 20.0,
    "traced_step_time": 100.0,
    "step_time": 110.0,
}


def _metrics(
    *,
    omit: tuple[str, ...] = (),
    overrides: dict[str, float] | None = None,
    worst: dict[str, float] | None = None,
) -> list[StepTimeMetric]:
    values = {**_COMPLETE_METRIC_VALUES, **(overrides or {})}
    worst_values = worst or {}
    return [
        _metric(name, value, worst=worst_values.get(name))
        for name, value in values.items()
        if name not in omit
    ]


def _timing(
    *,
    omit: tuple[str, ...] = (),
    **overrides: float,
) -> dict[str, float]:
    values = {**_COMPLETE_METRIC_VALUES, **overrides}
    return _canonical_timing(
        {name: value for name, value in values.items() if name not in omit}
    )


def _canonical_timing(values: dict[str, float]) -> dict[str, float]:
    """Keep direct dashboard fixtures within the canonical timing contract."""
    row = dict(values)
    if "input_wait" not in row or "traced_step_time" not in row:
        row.pop("step_time", None)
    else:
        row["step_time"] = row["input_wait"] + row["traced_step_time"]
    return row


def _payload(
    metrics: list[StepTimeMetric],
    *,
    per_rank_timing: dict[int, dict[str, float]] | None = None,
    clock: str = "cpu",
    freshness: LiveStepTimeFreshness = "live",
) -> LiveStepTimeResult:
    """Build a live result around one canonical Step Time window."""
    if per_rank_timing is None:
        row = {metric.metric: float(metric.median_total) for metric in metrics}
        if "input_wait" not in row or "traced_step_time" not in row:
            row.pop("step_time", None)
        else:
            row["step_time"] = row["input_wait"] + row["traced_step_time"]
        per_rank_timing = {0: row}
    else:
        per_rank_timing = {
            rank: _canonical_timing(values)
            for rank, values in per_rank_timing.items()
        }
    return live_result_from_window(
        window_from_rank_averages(
            per_rank_timing,
            clock="gpu" if clock == "gpu" else "cpu",
            expected_ranks=tuple(sorted(per_rank_timing)),
            metrics=metrics,
            steps_used=5,
        ),
        freshness=freshness,
    )


def _panel() -> dict:
    return {
        "win": _FakeText(),
        "summary": _FakeHtml(),
        "tree": _FakeHtml(),
        "_last_sig": None,
    }


_ROW_RE = re.compile(
    r"<td class='tml-tree-name'>(.*?)</td>"
    r"<td class='tml-num'>(.*?)</td><td class='tml-num'>(.*?)</td>"
)


def _rows(html: str) -> dict[str, tuple[str, str]]:
    """Map each row's label (tree glyphs stripped) to (value, share/scope)."""
    return {
        name.lstrip("├│└─ "): (value, extra)
        for name, value, extra in _ROW_RE.findall(html)
    }


def test_step_time_dashboard_hero_renders_sparse_metrics() -> None:
    # h2d and residual_proxy were never measured: the hero must still
    # update from the fresh sparse window instead of freezing on the last
    # complete view (issue #259).
    payload = _payload(_metrics(omit=("h2d", "residual_proxy")))
    panel = _panel()

    update_model_combined_section(panel, payload)

    assert panel["win"].text == "5 steps · CPU clock · partial: Residual"
    tree = _rows(panel["tree"].content)
    # H2D is occurrence-driven and absent, so it's plain measured-zero.
    assert tree["H2D"] == ("0.0 ms", "0%")
    # residual_proxy is genuinely unmeasured: n/a, never a fake 0%.
    assert tree["Residual"] == ("n/a", "—")
    # Measured phases use the canonical 110 ms Step Time denominator directly.
    assert tree["Input wait"] == ("10.0 ms", "9%")
    assert _rows(panel["summary"].content)["Step Time"][0] == "110 ms"


def test_step_time_dashboard_hero_measured_zero_is_not_partial() -> None:
    payload = _payload(
        _metrics(overrides={"h2d": 0.0, "residual_proxy": 30.0})
    )
    panel = _panel()

    update_model_combined_section(panel, payload)

    # A measured-zero H2D is complete coverage, not partial signals.
    assert panel["win"].text == "5 steps · CPU clock"
    assert _rows(panel["tree"].content)["H2D"] == ("0.0 ms", "0%")
    assert "n/a" not in panel["tree"].content


def test_step_time_dashboard_hero_absent_h2d_is_not_partial() -> None:
    # H2D events are occurrence-driven: a run with no host-to-device
    # copies emits none, which is complete coverage, not partial.
    payload = _payload(
        _metrics(omit=("h2d",), overrides={"residual_proxy": 30.0})
    )
    panel = _panel()

    update_model_combined_section(panel, payload)

    assert panel["win"].text == "5 steps · CPU clock"
    assert _rows(panel["tree"].content)["H2D"] == ("0.0 ms", "0%")


def test_step_time_dashboard_hero_step_time_only_extreme() -> None:
    payload = _payload([_metric("traced_step_time", 100.0)])
    panel = _panel()

    update_model_combined_section(panel, payload)

    # The inner envelope alone is not a canonical Step Time denominator.
    # Clear the existing card state rather than substituting it for Step Time.
    assert panel["win"].text == "CPU clock · Step Time unavailable"
    assert panel["summary"].content == ""
    assert panel["tree"].content == ""


def test_absent_optimizer_step_is_not_a_measured_zero() -> None:
    # optimizer_step is occurrence-driven, but "never observed in the whole
    # window" is absence, not an occurrence of zero: the canonical
    # availability rule drops a never-seen optimizer entirely, so the card
    # must not render it as a confident 0%.
    payload = _payload(_metrics(omit=("optimizer_step",)))
    panel = _panel()
    update_model_combined_section(panel, payload)

    tree = _rows(panel["tree"].content)
    assert tree["Optimizer"] == ("n/a", "—")
    # Compute includes its children, so it is unavailable too.
    assert tree["Compute"] == ("n/a", "—")
    # The available phases keep their own shares rather than rescaling.
    assert tree["Forward"] == ("20.0 ms", "18%")
    assert panel["win"].text == "5 steps · CPU clock · partial: Optimizer"


def test_dashboard_clears_when_input_wait_makes_step_time_unavailable() -> (
    None
):
    # residual_proxy can survive an unmeasured input wait, but canonical Step
    # Time cannot. The dashboard must not substitute the inner envelope.
    payload = _payload(
        _metrics(
            omit=("input_wait", "step_time"),
            overrides={"residual_proxy": 10.0},
        )
    )
    panel = _panel()
    update_model_combined_section(panel, payload)

    assert panel["win"].text == "CPU clock · Step Time unavailable"
    assert panel["summary"].content == ""

    # Control twin: canonical Step Time restores the existing card.
    with_wait = _payload(_metrics(overrides={"residual_proxy": 10.0}))
    control = _panel()
    update_model_combined_section(control, with_wait)
    assert _rows(control["tree"].content)["Residual"] == ("10.0 ms", "9%")


def test_partial_rank_coverage_includes_step_time() -> None:
    # step_time is not a phase, but it is the denominator every share is
    # computed against, so a rank missing it makes the window incomplete
    # exactly as the engine reports it.
    payload = _payload(
        _metrics(),
        per_rank_timing={
            0: _timing(),
            1: _timing(omit=("residual_proxy", "traced_step_time")),
        },
    )
    panel = _panel()
    update_model_combined_section(panel, payload)

    assert panel["win"].text == (
        "5 steps · CPU clock · partial: Residual, Step Time"
    )


def test_step_time_dashboard_hero_dark_input_wait_is_not_measured_zero() -> (
    None
):
    # #88/#135 shape: the input stream goes dark while everything else keeps
    # reporting. The card must not derive Step Time from the remaining phases.
    dark_input_wait = _payload(_metrics(omit=("input_wait", "step_time")))
    panel = _panel()
    update_model_combined_section(panel, dark_input_wait)

    assert panel["win"].text == "CPU clock · Step Time unavailable"
    assert panel["tree"].content == ""

    # Control twin: input_wait genuinely measured as 0ms is a real 0%,
    # distinct from the unmeasured case above.
    measured_zero_input_wait = _payload(
        _metrics(overrides={"input_wait": 0.0})
    )
    control_panel = _panel()
    update_model_combined_section(control_panel, measured_zero_input_wait)
    assert _rows(control_panel["tree"].content)["Input wait"] == (
        "0.0 ms",
        "0%",
    )
    assert control_panel["win"].text == "5 steps · CPU clock"


def test_step_time_dashboard_hero_dead_run_shows_expired() -> None:
    # The CLI sibling distinguishes "no data yet" from "had data, now
    # expired" via live-session freshness; the card must do the same
    # instead of freezing on the last complete view forever.
    good = _payload(_metrics())
    panel = _panel()
    update_model_combined_section(panel, good)
    assert panel["win"].text == "5 steps · CPU clock"
    assert "110 ms" in panel["summary"].content

    # A never-had-data payload (cold start) must NOT clear the panel --
    # there is nothing to clear yet.
    cold_start = live_result_from_window(StepTimeWindow(), freshness="cold")
    cold_panel = _panel()
    update_model_combined_section(cold_panel, cold_start)
    assert cold_panel["win"].text == ""

    # TTL expiry must visibly clear rather than keep showing "good".
    expired = live_result_from_window(StepTimeWindow(), freshness="expired")
    update_model_combined_section(panel, expired)

    assert panel["win"].text == "window expired"
    assert panel["summary"].content == ""
    assert panel["tree"].content == ""


def test_step_time_dashboard_hero_recovered_phase_restores_its_row() -> None:
    sparse = _payload(_metrics(omit=("forward",)))
    panel = _panel()
    update_model_combined_section(panel, sparse)

    assert _rows(panel["tree"].content)["Forward"] == ("n/a", "—")
    assert panel["win"].text == "5 steps · CPU clock · partial: Forward"

    # Forward starts reporting again.
    update_model_combined_section(panel, _payload(_metrics()))

    assert _rows(panel["tree"].content)["Forward"] == ("20.0 ms", "18%")
    assert panel["win"].text == "5 steps · CPU clock"


def test_step_time_dashboard_hero_flags_partial_rank_coverage() -> None:
    # Aggregate presence is not coverage. Backward is measured on rank 0
    # and absent on rank 1, so the canonical diagnosis calls the window
    # INCOMPLETE; the card must not report it as fully covered just
    # because the aggregate metric exists.
    payload = _payload(
        _metrics(),
        per_rank_timing={
            0: _timing(omit=("residual_proxy",)),
            1: _timing(omit=("backward", "residual_proxy")),
        },
    )
    panel = _panel()
    update_model_combined_section(panel, payload)

    assert panel["win"].text == (
        "5 steps · CPU clock · partial: Backward, Residual"
    )

    # Control twin: every rank measured everything -> not partial.
    symmetric = _payload(
        payload.analysis.window.metrics,
        per_rank_timing={rank: _timing() for rank in (0, 1)},
    )
    control = _panel()
    update_model_combined_section(control, symmetric)
    assert control["win"].text == "5 steps · CPU clock"


def test_step_time_dashboard_hero_clears_when_step_time_missing() -> None:
    # Stateful: a complete window is drawn first, then a window arrives with
    # no canonical Step Time. The card must clear rather than leave the
    # previous values standing as if they were current.
    panel = _panel()
    update_model_combined_section(panel, _payload(_metrics()))
    assert "110 ms" in panel["summary"].content

    no_envelope = _payload(
        _metrics(
            omit=(
                "h2d",
                "backward",
                "optimizer_step",
                "residual_proxy",
                "traced_step_time",
                "step_time",
            )
        )
    )
    update_model_combined_section(panel, no_envelope)

    assert panel["win"].text == "CPU clock · Step Time unavailable"
    assert panel["summary"].content == ""
    assert panel["tree"].content == ""


def test_multi_rank_summary_shows_median_and_worst_rank() -> None:
    payload = _payload(
        _metrics(worst={"step_time": 210.0}),
        per_rank_timing={0: _timing(), 1: _timing()},
        clock="gpu",
    )
    panel = _panel()

    update_model_combined_section(panel, payload)

    summary = _rows(panel["summary"].content)
    assert summary["Median"] == ("110 ms", "Across ranks")
    assert summary["Worst"] == ("210 ms", "Rank 0")
    assert panel["win"].text == "5 steps · GPU clock"
    assert "Phase breakdown · Rank 0" in panel["tree"].content


def test_single_rank_summary_shows_one_step_time() -> None:
    panel = _panel()

    update_model_combined_section(panel, _payload(_metrics()))

    assert _rows(panel["summary"].content) == {"Step Time": ("110 ms", "")}
    assert "Phase breakdown<" in panel["tree"].content
    assert "Rank" not in panel["tree"].content


def test_tree_reads_one_rank_and_uses_its_step_time_for_shares() -> None:
    metrics = _metrics(
        omit=("h2d",),
        overrides={
            "input_wait": 50.0,
            "forward": 45.0,
            "backward": 15.0,
            "optimizer_step": 7.5,
            "residual_proxy": 32.5,
        },
    )
    payload = _payload(
        metrics,
        per_rank_timing={
            0: _timing(
                omit=("h2d",),
                input_wait=0.0,
                forward=10.0,
                backward=20.0,
                optimizer_step=10.0,
                residual_proxy=60.0,
            ),
            1: _timing(
                omit=("h2d",),
                input_wait=100.0,
                forward=80.0,
                backward=10.0,
                optimizer_step=5.0,
                residual_proxy=5.0,
            ),
        },
    )
    panel = _panel()

    update_model_combined_section(panel, payload)

    # Rank 0's own 100 ms Step Time, not the across-rank 110 ms median.
    assert "Phase breakdown · Rank 0" in panel["tree"].content
    assert _rows(panel["tree"].content) == {
        "Step Time": ("100.0 ms", "100%"),
        "Input wait": ("0.0 ms", "0%"),
        "Compute": ("40.0 ms", "40%"),
        "Forward": ("10.0 ms", "10%"),
        "Backward": ("20.0 ms", "20%"),
        "Optimizer": ("10.0 ms", "10%"),
        "H2D": ("0.0 ms", "0%"),
        "Residual": ("60.0 ms", "60%"),
    }


@pytest.mark.parametrize("step_ms", (0.0, -1.0))
def test_non_positive_step_time_shows_no_shares(step_ms: float) -> None:
    payload = _payload(
        _metrics(),
        per_rank_timing={0: _timing(input_wait=step_ms, traced_step_time=0.0)},
    )
    panel = _panel()

    update_model_combined_section(panel, payload)

    tree = panel["tree"].content
    assert _rows(tree)["Step Time"] == (f"{step_ms:.1f} ms", "")
    assert "%" not in tree
    assert "Step Time unavailable for phase shares" in tree
    assert "110 ms" in panel["summary"].content


def test_disjoint_rank_populations_keep_the_summary() -> None:
    payload = _payload(
        _metrics(
            omit=("h2d", "residual_proxy"),
            overrides={
                "forward": 40.0,
                "backward": 40.0,
                "optimizer_step": 10.0,
            },
        ),
        per_rank_timing={
            0: _timing(
                omit=("h2d", "backward", "optimizer_step", "residual_proxy"),
                forward=40.0,
            ),
            1: _timing(
                omit=(
                    "input_wait",
                    "h2d",
                    "forward",
                    "residual_proxy",
                    "traced_step_time",
                ),
                backward=40.0,
                optimizer_step=10.0,
                traced_step_time=120.0,
            ),
        },
    )
    panel = _panel()

    update_model_combined_section(panel, payload)

    assert "no coherent phase cohort" in panel["tree"].content
    assert _rows(panel["tree"].content) == {}
    assert _rows(panel["summary"].content)["Median"][0] == "110 ms"


def test_card_builds_without_ribbon_or_verdict() -> None:
    panel = build_model_combined_section()

    assert set(panel) == {"win", "summary", "tree", "_last_sig"}
    assert panel["win"].text == "waiting for steps"


def test_sqlite_window_has_one_share_across_live_and_summary_consumers(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "step-time-pipeline.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE step_time_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_timestamp_unix_ns INTEGER NOT NULL,
                rank INTEGER,
                global_rank INTEGER,
                local_rank INTEGER,
                world_size INTEGER,
                local_world_size INTEGER,
                node_rank INTEGER,
                hostname TEXT,
                timestamp_unix_ns INTEGER,
                seq INTEGER,
                step INTEGER,
                events_json TEXT NOT NULL
            );
            """)
        conn.execute("""
            CREATE TABLE runtime_environment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_strategy TEXT
            );
            """)
        conn.execute(
            "INSERT INTO runtime_environment(training_strategy) "
            "VALUES ('ddp');"
        )
        values = {
            "_traceml_internal:dataloader_next": 100.0,
            "_traceml_internal:forward_time": 30.0,
            "_traceml_internal:backward_time": 30.0,
            "_traceml_internal:optimizer_step": 30.0,
            "_traceml_internal:step_time": 100.0,
        }
        events_json = json.dumps(
            {
                name: {
                    "cpu": {
                        "duration_ms": value,
                        "cpu_ms": value,
                        "gpu_ms": None,
                        "n_calls": 1,
                    }
                }
                for name, value in values.items()
            }
        )
        conn.executemany(
            """
            INSERT INTO step_time_samples(
                observed_timestamp_unix_ns, rank, global_rank, local_rank, world_size,
                local_world_size, node_rank, hostname, timestamp_unix_ns,
                seq, step, events_json
            ) VALUES (?, 0, 0, 0, 1, 1, 0, 'worker-0', ?, ?, ?, ?);
            """,
            [
                (step, step * 1_000_000_000, step, step, events_json)
                for step in range(1, 31)
            ],
        )
        conn.commit()
    finally:
        conn.close()

    result = LiveStepTimeSession(
        str(db_path),
        request=StepTimeLoadRequest(window_size=30, lookback_factor=4),
    ).refresh()
    window = result.analysis.window
    assert rank_average(window, 0).residual_ms == pytest.approx(10.0)
    assert window.residual_share == pytest.approx(0.05)

    summary = StepTimeSummarySection().build(str(db_path)).payload
    summary_metrics = summary["groups"]["rows"]["0"]["metrics"]
    assert summary_metrics["residual_ms"] == pytest.approx(10.0)
    assert summary_metrics["h2d_ms"] is None

    renderer = StepTimeRenderer(
        LiveStepTimeSession(
            str(db_path),
            request=StepTimeLoadRequest(
                window_size=30,
                lookback_factor=4,
            ),
        )
    )
    console = Console(record=True, width=140, color_system=None)
    console.print(renderer.get_panel_renderable())
    assert "5.0%" in console.export_text()

    diagnostics = build_model_diagnostics_payload(
        step_time_window=window,
        step_time_diagnosis=result.analysis.diagnosis.primary,
        step_memory_metrics=(),
    )
    step_item = next(
        item for item in diagnostics.items if item.source == "step_time"
    )
    assert step_item.evidence["residual"] == "5.0%"

    panel = _panel()
    update_model_combined_section(panel, result)
    assert _rows(panel["tree"].content)["Residual"] == ("10.0 ms", "5%")


def test_diagnostics_rail_incomplete_data_is_neutral() -> None:
    from traceml_ai.aggregator.display_drivers.nicegui_sections import (
        model_diagnostics_section as mds,
    )

    assert (
        mds._row_sev({"severity": "info", "kind": "INCOMPLETE_DATA"})
        == "neutral"
    )
    assert mds._row_sev({"severity": "info", "kind": "BALANCED"}) == "healthy"


def test_diagnostics_rail_row_renders_engine_action() -> None:
    # Issue #357: the rail dropped the engine's action, so the dashboard
    # showed a verdict with no next step while the terminal printed one.
    from traceml_ai.aggregator.display_drivers.nicegui_sections import (
        model_diagnostics_section as mds,
    )

    row = mds._row_html(
        {
            "source": "step_time",
            "status": "INPUT-BOUND",
            "reason": "Input wait is 80.0% of the typical CPU Step Time.",
            "action": "Increase workers, prefetch, or storage throughput.",
        }
    )

    assert "Next: Increase workers, prefetch, or storage throughput." in row


def test_diagnostics_rail_row_renders_action_and_note() -> None:
    from traceml_ai.aggregator.display_drivers.nicegui_sections import (
        model_diagnostics_section as mds,
    )

    row = mds._row_html(
        {
            "source": "step_memory",
            "status": "MEMORY-CREEP",
            "reason": "Allocated memory grows across the window.",
            "action": "Check for retained tensors between steps.",
            "note": "peak 12.4 GB on rank 3",
        }
    )

    assert "Next: Check for retained tensors between steps." in row
    assert "peak 12.4 GB on rank 3" in row


@pytest.mark.parametrize("action", (None, "", "   "))
def test_diagnostics_rail_row_omits_next_without_action(
    action: str | None,
) -> None:
    from traceml_ai.aggregator.display_drivers.nicegui_sections import (
        model_diagnostics_section as mds,
    )

    item = {
        "source": "step_time",
        "status": "NO DATA",
        "reason": "Waiting for the first complete window.",
    }
    if action is not None:
        item["action"] = action

    row = mds._row_html(item)

    assert "Next:" not in row
    assert "Waiting for the first complete window." in row


def test_diagnostics_rail_row_truncates_action_like_reason() -> None:
    from traceml_ai.aggregator.display_drivers.nicegui_sections import (
        model_diagnostics_section as mds,
    )

    row = mds._row_html(
        {
            "source": "step_time",
            "status": "INPUT-BOUND",
            "reason": "R" * 300,
            "action": "A" * 300,
            "note": "N" * 300,
        }
    )

    for char in ("R", "A", "N"):
        assert char * 127 + "…" in row
        assert char * 128 not in row


def test_diagnostics_rail_row_never_truncates_inside_an_html_entity() -> None:
    """Cutting escaped text can split ``&amp;`` into literal ``&a``."""
    from traceml_ai.aggregator.display_drivers.nicegui_sections import (
        model_diagnostics_section as mds,
    )

    # Places the '&' where its 5-character entity would straddle the cut.
    long_text = "x" * 125 + "& more text that runs past the cap" + "z" * 40

    row = mds._row_html(
        {
            "source": "step_time",
            "status": "INPUT-BOUND",
            "reason": long_text,
            "action": long_text,
            "note": long_text,
        }
    )

    assert "&amp;" in row
    for broken in ("&a…", "&am…", "&amp…"):
        assert broken not in row


def test_diagnostics_rail_next_line_matches_cli_diagnosis() -> None:
    """One engine verdict reads the same on the terminal and the rail."""
    from traceml_ai.aggregator.display_drivers.nicegui_sections import (
        model_diagnostics_section as mds,
    )

    result = _payload(
        _metrics(
            overrides={
                "input_wait": 80.0,
                "h2d": 0.0,
                "forward": 5.0,
                "backward": 10.0,
                "optimizer_step": 4.0,
                "residual_proxy": 1.0,
                "traced_step_time": 20.0,
            }
        )
    )
    diagnosis = result.analysis.diagnosis.primary
    assert diagnosis.kind == "INPUT_BOUND"
    assert diagnosis.action

    payload = build_model_diagnostics_payload(
        step_time_window=result.analysis.window,
        step_time_diagnosis=diagnosis,
        step_memory_metrics=(),
    ).to_dict()
    item = next(
        entry for entry in payload["items"] if entry["source"] == "step_time"
    )

    assert diagnosis.action in format_cli_diagnosis(diagnosis)
    assert f"Next: {diagnosis.action}" in mds._row_html(item)


def test_empty_diagnostics_payload_clears_stale_ui_state() -> None:
    from traceml_ai.aggregator.display_drivers.nicegui_sections import (
        model_diagnostics_section as mds,
    )

    panel = {
        "overall": _FakeText(),
        "body": _FakeHtml(),
        "hint": _FakeText(),
    }
    panel["overall"].text = "CRITICAL"
    panel["body"].content = "stale diagnosis"

    mds.update_model_diagnostics_section(panel, {"items": []})

    assert panel["overall"].text == "NO DATA"
    assert panel["body"].content == ""
    assert panel["hint"].text == "Waiting for diagnostics"
    assert theme.SEV["neutral"] in panel["overall"].styles[-1]


@pytest.mark.parametrize(
    ("severity", "bucket"),
    (("crit", "crit"), ("warn", "warn"), ("info", "neutral")),
)
def test_diagnostics_payload_fills_the_rail_in_the_engine_colour(
    severity: str, bucket: str
) -> None:
    """The pill reads the engine's overall severity and nothing else."""
    from traceml_ai.aggregator.display_drivers.nicegui_sections import (
        model_diagnostics_section as mds,
    )

    panel = {
        "overall": _FakeText(),
        "body": _FakeHtml(),
        "hint": _FakeText(),
    }
    mds.update_model_diagnostics_section(panel, {"items": []})
    mds.update_model_diagnostics_section(
        panel,
        {
            "overall_severity": severity,
            "items": [
                {
                    "source": "step_time",
                    "status": "INPUT-BOUND",
                    "severity": severity,
                    "reason": "Input wait dominates the step.",
                }
            ],
        },
    )

    assert panel["overall"].text == severity.upper()
    assert theme.SEV[bucket] in panel["overall"].styles[-1]
    assert panel["hint"].text == ""
    assert "INPUT-BOUND" in panel["body"].content


def test_diagnostics_pill_follows_overall_not_the_first_item() -> None:
    """Items keep registry order, so the first one need not be the worst."""
    from traceml_ai.aggregator.display_drivers.nicegui_sections import (
        model_diagnostics_section as mds,
    )

    panel = {
        "overall": _FakeText(),
        "body": _FakeHtml(),
        "hint": _FakeText(),
    }
    mds.update_model_diagnostics_section(
        panel,
        {
            "overall_severity": "warn",
            "items": [
                {
                    "source": "step_memory",
                    "status": "BALANCED",
                    "severity": "info",
                    "reason": "Memory is steady.",
                },
                {
                    "source": "step_time",
                    "status": "INPUT-BOUND",
                    "severity": "warn",
                    "reason": "Input wait dominates the step.",
                },
            ],
        },
    )

    assert panel["overall"].text == "WARN"
    assert theme.SEV["warn"] in panel["overall"].styles[-1]
