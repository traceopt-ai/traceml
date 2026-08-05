from __future__ import annotations

import json
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
    update_model_combined_section,
    update_step_verdict,
)
from traceml_ai.diagnostics.model_diagnostics import (
    build_model_diagnostics_payload,
)
from traceml_ai.renderers.step_time.renderer import StepTimeRenderer
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


class _FakeSegment:
    """Style double that MERGES like NiceGUI's ``Element.style()``.

    An append-only list of style strings would let a stale declaration
    (e.g. an unmeasured phase's hatch) survive a later width-only update
    while every assertion still passed, because assertions read the last
    call instead of the resulting style. Merging into a dict is what
    makes that class of bug visible to a test at all. Pinned against the
    real widget by ``test_fake_segment_matches_nicegui_style_merge``.
    """

    def __init__(self) -> None:
        self.styles: list[str] = []
        self.declarations: dict[str, str] = {}

    def style(self, value: str) -> "_FakeSegment":
        self.styles.append(value)
        for part in value.split(";"):
            if ":" not in part:
                continue
            name, _, val = part.partition(":")
            self.declarations[name.strip()] = val.strip()
        return self

    @property
    def background(self) -> str:
        return self.declarations.get("background", "")

    @property
    def width(self) -> str:
        return self.declarations.get("width", "")

    @property
    def hatched(self) -> bool:
        return "repeating-linear-gradient" in self.background


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
        "seg_divs": [_FakeSegment() for _ in theme.PHASES],
        "seg_labs": [_FakeText() for _ in theme.PHASES],
        "win": _FakeText(),
        "verdict": _FakeText(),
        "kpis": {
            "median": _FakeHtml(),
            "worst": _FakeHtml(),
            "gap": _FakeHtml(),
            "residual": _FakeHtml(),
            "rank": _FakeHtml(),
        },
        "_last_sig": None,
    }


def test_step_time_dashboard_hero_renders_sparse_metrics() -> None:
    # h2d and residual_proxy were never measured: the hero must still
    # update from the fresh sparse window instead of freezing on the last
    # complete view (issue #259).
    diagnosis_metrics = _metrics(omit=("h2d", "residual_proxy"))
    payload = _payload(diagnosis_metrics)
    panel = _panel()

    update_model_combined_section(panel, payload)

    # The panel updated (no freeze): window label reflects the sparse
    # window and names the missing signal.
    assert (
        panel["win"].text
        == "5 aligned steps · CPU selected · representative r0 · partial: "
        "RESIDUAL"
    )
    # H2D is occurrence-driven and absent, so it's plain measured-zero.
    # residual_proxy is genuinely unmeasured, so it has no visual share.
    # Measured phases use the canonical 110 ms Step Time denominator directly.
    keys = [key for _, key, _ in theme.PHASES]
    by_key = dict(zip(keys, panel["seg_divs"]))
    by_lab = dict(zip(keys, panel["seg_labs"]))
    assert by_key["h2d"].width == "0.000%"
    residual_seg = by_key["residual_proxy"]
    assert residual_seg.width == "0%"
    assert residual_seg.background == "transparent"
    # Missing signals are named in the existing window metadata instead of
    # receiving a fabricated proportional ribbon segment.
    assert by_lab["residual_proxy"].text == ""
    assert by_key["input_wait"].width == "9.091%"
    # An underivable residual shows n/a, never a fake 0%.
    assert "n/a" in panel["kpis"]["residual"].content
    assert panel["kpis"]["median"].content.startswith("110")


def test_step_time_dashboard_hero_measured_zero_is_not_partial() -> None:
    diagnosis_metrics = _metrics(
        overrides={"h2d": 0.0, "residual_proxy": 30.0}
    )
    payload = _payload(diagnosis_metrics)
    panel = _panel()

    update_model_combined_section(panel, payload)

    # A measured-zero H2D is complete coverage, not partial signals.
    assert panel["win"].text == (
        "5 aligned steps · CPU selected · representative r0"
    )
    assert "n/a" not in panel["kpis"]["residual"].content


def test_step_time_dashboard_hero_absent_h2d_is_not_partial() -> None:
    # H2D events are occurrence-driven: a run with no host-to-device
    # copies emits none, which is complete coverage, not partial.
    diagnosis_metrics = _metrics(
        omit=("h2d",),
        overrides={"residual_proxy": 30.0},
    )
    payload = _payload(diagnosis_metrics)
    panel = _panel()

    update_model_combined_section(panel, payload)

    assert panel["win"].text == (
        "5 aligned steps · CPU selected · representative r0"
    )
    keys = [key for _, key, _ in theme.PHASES]
    by_key = dict(zip(keys, panel["seg_divs"]))
    assert by_key["h2d"].width == "0.000%"


def test_step_time_dashboard_hero_step_time_only_extreme() -> None:
    payload = _payload([_metric("traced_step_time", 100.0)])
    panel = _panel()

    update_model_combined_section(panel, payload)

    # The inner envelope alone is not a canonical Step Time denominator.
    # Clear the existing card state rather than substituting it for Step Time.
    assert panel["win"].text == "CPU selected · Step Time unavailable"
    assert all(seg.width == "0%" for seg in panel["seg_divs"])
    assert all(
        kpi.content == theme.kval("—") for kpi in panel["kpis"].values()
    )


def test_absent_optimizer_step_is_not_a_measured_zero() -> None:
    # Narrow guard for the predicate itself. optimizer_step is
    # occurrence-driven, but "never observed in the whole window" is
    # absence, not an occurrence of zero: the canonical availability rule
    # drops a never-seen optimizer entirely, so the card must not render
    # it as a confident 0%.
    payload = _payload(_metrics(omit=("optimizer_step",)))
    panel = _panel()
    update_model_combined_section(panel, payload)

    keys = [key for _, key, _ in theme.PHASES]
    by_key = dict(zip(keys, panel["seg_divs"]))
    assert by_key["optimizer_step"].width == "0%"
    assert by_key["optimizer_step"].background == "transparent"
    assert (
        panel["win"].text
        == "5 aligned steps · CPU selected · representative r0 · partial: "
        "OPT"
    )


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

    assert panel["win"].text == "CPU selected · Step Time unavailable"
    assert all(
        kpi.content == theme.kval("—") for kpi in panel["kpis"].values()
    )

    # Control twin: canonical Step Time restores the existing card.
    with_wait = _payload(_metrics(overrides={"residual_proxy": 10.0}))
    control = _panel()
    update_model_combined_section(control, with_wait)
    assert "n/a" not in control["kpis"]["residual"].content


def test_partial_rank_coverage_includes_step_time() -> None:
    # step_time is not a ribbon phase, but it is the envelope every share
    # is computed against, so a rank missing it makes the window
    # incomplete exactly as the engine reports it.
    payload = _payload(
        _metrics(),
        per_rank_timing={
            0: _timing(),
            1: _timing(omit=("residual_proxy", "traced_step_time")),
        },
    )
    panel = _panel()
    update_model_combined_section(panel, payload)

    assert (
        panel["win"].text
        == "5 aligned steps · CPU selected · representative r0 · partial: "
        "RESIDUAL,STEP TIME"
    )


def test_step_time_dashboard_hero_dark_input_wait_is_not_measured_zero() -> (
    None
):
    # #88/#135 shape: the input stream goes dark while everything else keeps
    # reporting. The card must not derive Step Time from the remaining phases.
    dark_input_wait = _payload(_metrics(omit=("input_wait", "step_time")))
    panel = _panel()
    update_model_combined_section(panel, dark_input_wait)

    assert panel["win"].text == "CPU selected · Step Time unavailable"
    assert all(seg.width == "0%" for seg in panel["seg_divs"])

    # Control twin: input_wait genuinely measured as 0ms must render as
    # a real 0%-width segment, distinct from the unmeasured case above.
    measured_zero_input_wait = _payload(
        _metrics(overrides={"input_wait": 0.0})
    )
    control_panel = _panel()
    update_model_combined_section(control_panel, measured_zero_input_wait)
    keys = [key for _, key, _ in theme.PHASES]
    control_by_key = dict(zip(keys, control_panel["seg_divs"]))
    assert control_by_key["input_wait"].width == "0.000%"
    assert (
        control_panel["win"].text
        == "5 aligned steps · CPU selected · representative r0"
    )


def test_step_time_dashboard_hero_dead_run_shows_expired() -> None:
    # The CLI sibling distinguishes "no data yet" from "had data, now
    # expired" via live-session freshness; the compatibility projection must
    # do the same instead of freezing on the last complete view forever.
    good = _payload(_metrics())
    panel = _panel()
    update_model_combined_section(panel, good)
    assert panel["win"].text == (
        "5 aligned steps · CPU selected · representative r0"
    )
    assert panel["kpis"]["median"].content.startswith("110")

    # A never-had-data payload (cold start) must NOT clear the panel --
    # there is nothing to clear yet.
    cold_start = live_result_from_window(
        StepTimeWindow(),
        freshness="cold",
    )
    cold_panel = _panel()
    update_model_combined_section(cold_panel, cold_start)
    assert cold_panel["win"].text == ""

    # TTL expiry must visibly clear rather than keep showing "good".
    expired = live_result_from_window(
        StepTimeWindow(),
        freshness="expired",
    )
    update_model_combined_section(panel, expired)

    assert panel["win"].text == "window expired"
    assert all(seg.width == "0%" for seg in panel["seg_divs"])
    assert all(sl.text == "" for sl in panel["seg_labs"])
    assert all(
        kpi.content == theme.kval("—") for kpi in panel["kpis"].values()
    )


def test_fake_segment_matches_nicegui_style_merge() -> None:
    """Pin the double against the real widget.

    ``Element.style()`` MERGES declarations, so a background set while a
    phase is unmeasured must be explicitly cleared. A double
    that only appended style strings could never express that, which is
    how a stale segment could reach review: the test could not fail.
    """
    from nicegui import ui

    real = ui.element("div")
    fake = _FakeSegment()
    for call in (
        "background:#1976d2; width:0%;",
        "width:0%; background:transparent;",
        "width:33.000%; background:#1976d2;",
    ):
        real.style(call)
        fake.style(call)

    assert dict(real._style) == fake.declarations
    # The production path must restate the background when a signal changes.
    assert fake.background == "#1976d2"


def test_step_time_dashboard_hero_recovered_phase_restores_its_segment() -> (
    None
):
    # Stateful sparse -> complete. NiceGUI merges styles, so a cleared phase
    # must restore its solid background and real width when it recovers.
    sparse = _payload(_metrics(omit=("forward",)))
    panel = _panel()
    update_model_combined_section(panel, sparse)

    keys = [key for _, key, _ in theme.PHASES]
    by_key = dict(zip(keys, panel["seg_divs"]))
    assert by_key["forward"].width == "0%"
    assert by_key["forward"].background == "transparent"
    assert (
        panel["win"].text
        == "5 aligned steps · CPU selected · representative r0 · partial: "
        "FWD"
    )

    # Forward starts reporting again.
    complete = _payload(_metrics())
    update_model_combined_section(panel, complete)

    forward_color = dict((k, c) for _, k, c in theme.PHASES)["forward"]
    assert by_key["forward"].background == forward_color
    assert by_key["forward"].width != "0%"
    assert (
        panel["win"].text
        == "5 aligned steps · CPU selected · representative r0"
    )


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

    assert (
        panel["win"].text
        == "5 aligned steps · CPU selected · representative r0 · partial: "
        "BWD,RESIDUAL"
    )

    # Control twin: every rank measured everything -> not partial.
    window = payload.analysis.window
    symmetric = _payload(
        window.metrics,
        per_rank_timing={rank: _timing() for rank in (0, 1)},
    )
    control = _panel()
    update_model_combined_section(control, symmetric)
    assert (
        control["win"].text
        == "5 aligned steps · CPU selected · representative r0"
    )


def test_step_time_dashboard_hero_clears_when_step_time_missing() -> None:
    # Stateful: a COMPLETE window is drawn first, then a window arrives
    # with no canonical Step Time. Without that denominator, so
    # every share would be invented -- the card must clear rather than
    # leave the previous ribbon and KPIs standing as if they were current.
    complete = _payload(_metrics())
    panel = _panel()
    update_model_combined_section(panel, complete)
    assert panel["win"].text == (
        "5 aligned steps · CPU selected · representative r0"
    )
    assert panel["kpis"]["median"].content.startswith("110")

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

    assert panel["win"].text == "CPU selected · Step Time unavailable"
    assert all(seg.width == "0%" for seg in panel["seg_divs"])
    assert all(
        kpi.content == theme.kval("—") for kpi in panel["kpis"].values()
    )


def test_step_time_dashboard_hero_uses_diagnosis_metrics() -> None:
    assert theme.PHASES[0][:2] == ("IW", "input_wait")

    # Self-consistent window shape: phases sum to canonical Step Time.
    diagnosis_metrics = _metrics(worst={"step_time": 210.0})
    payload = _payload(diagnosis_metrics, clock="gpu")
    panel = _panel()

    update_model_combined_section(panel, payload)

    assert panel["seg_labs"][0].text == "IW"
    # input_wait 10 of the 110 ms iteration envelope.
    assert panel["seg_divs"][0].width == "9.091%"
    assert panel["win"].text == (
        "5 aligned steps · GPU selected · representative r0"
    )
    assert panel["kpis"]["median"].content.startswith("110")
    assert panel["kpis"]["worst"].content.startswith("210")
    assert not panel["kpis"]["median"].content.startswith("20")


def test_ribbon_selects_one_real_representative_rank() -> None:
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

    keys = [key for _, key, _ in theme.PHASES]
    segments = dict(zip(keys, panel["seg_divs"]))
    assert "representative r0" in panel["win"].text
    assert segments["forward"].width == "10.000%"
    assert segments["residual_proxy"].width == "60.000%"


def test_disjoint_rank_populations_clear_only_the_ribbon() -> None:
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

    assert "no coherent phase cohort" in panel["win"].text
    assert all(segment.width == "0%" for segment in panel["seg_divs"])
    assert panel["kpis"]["median"].content.startswith("110")
    assert panel["kpis"]["gap"].content != theme.kval("—")


def test_sqlite_window_has_one_share_across_live_and_summary_consumers(
    tmp_path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "step-time-pipeline.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE step_time_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                recv_ts_ns INTEGER NOT NULL,
                rank INTEGER,
                global_rank INTEGER,
                local_rank INTEGER,
                world_size INTEGER,
                local_world_size INTEGER,
                node_rank INTEGER,
                hostname TEXT,
                sample_ts_s REAL,
                seq INTEGER,
                step INTEGER,
                events_json TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE runtime_environment (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                training_strategy TEXT
            );
            """
        )
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
                recv_ts_ns, rank, global_rank, local_rank, world_size,
                local_world_size, node_rank, hostname, sample_ts_s,
                seq, step, events_json
            ) VALUES (?, 0, 0, 0, 1, 1, 0, 'worker-0', ?, ?, ?, ?);
            """,
            [
                (step, float(step), step, step, events_json)
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

    summary = StepTimeSummarySection(max_rows=30).build(str(db_path)).payload
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
    assert panel["kpis"]["residual"].content == theme.kval("5", "%")


def test_diagnostics_rail_incomplete_data_is_neutral() -> None:
    from traceml_ai.aggregator.display_drivers.nicegui_sections import (
        model_diagnostics_section as mds,
    )

    assert (
        mds._row_sev({"severity": "info", "kind": "INCOMPLETE_DATA"})
        == "neutral"
    )
    assert mds._row_sev({"severity": "info", "kind": "BALANCED"}) == "healthy"


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


def test_empty_diagnostics_payload_clears_stale_hero_verdict() -> None:
    panel = _panel()
    panel["verdict"].text = "COMPUTE-BOUND"

    update_step_verdict(panel, {"items": []})

    assert panel["verdict"].text == "NO DATA"
