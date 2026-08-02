from __future__ import annotations

import json
import sqlite3

import pytest
from rich.console import Console

pytest.importorskip("nicegui")

from traceml_ai.aggregator.display_drivers.nicegui_sections import theme
from traceml_ai.aggregator.display_drivers.nicegui_sections.model_combined_section import (
    _UNMEASURED_SLIVER_PCT,
    update_model_combined_section,
)
from traceml_ai.diagnostics.model_diagnostics import (
    build_model_diagnostics_payload,
)
from traceml_ai.diagnostics.step_time import LIVE_STEP_TIME_POLICY
from traceml_ai.renderers.step_time.compute import StepCombinedComputer
from traceml_ai.renderers.step_time.renderer import StepCombinedRenderer
from traceml_ai.step_time.model import (
    StepTimeCoverage,
    StepTimeMetric,
    StepTimeResult,
    StepTimeSummary,
)
from traceml_ai.reporting.sections.step_time.loader import (
    load_step_time_section_data,
)
from traceml_ai.utils.step_time_window import (
    StepTimeWindow,
    diagnose_step_time_window,
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
        clock="gpu",
        series=None,
        summary=StepTimeSummary(
            window_size=5,
            steps_used=5,
            median_total=value,
            worst_total=worst if worst is not None else value,
            worst_rank=0,
            skew_ratio=0.0,
            skew_pct=0.0,
        ),
        coverage=StepTimeCoverage(
            expected_steps=5,
            steps_used=5,
            completed_step=5,
            world_size=1,
            ranks_present=1,
            incomplete=False,
        ),
    )


def _payload(
    metrics: list[StepTimeMetric],
    *,
    per_rank_timing: dict[int, dict[str, float]] | None = None,
    clock: str = "cpu",
    had_ok: bool = False,
) -> StepTimeResult:
    """Build a live result around one canonical Step Time window."""
    if per_rank_timing is None:
        row = {
            metric.metric: float(metric.summary.median_total)
            for metric in metrics
        }
        if "input_wait" in row and "step_time" in row:
            row["total_step"] = row["input_wait"] + row["step_time"]
        per_rank_timing = {0: row}
    return StepTimeResult(
        window=StepTimeWindow(
            clock="gpu" if clock == "gpu" else "cpu",
            expected_ranks=tuple(sorted(per_rank_timing)),
            per_rank_timing=per_rank_timing,
            metrics=metrics,
        ),
        had_ok=had_ok,
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
    diagnosis_metrics = [
        _metric("input_wait", 10.0),
        _metric("forward", 20.0),
        _metric("backward", 30.0),
        _metric("optimizer_step", 20.0),
        _metric("step_time", 100.0),
    ]
    payload = _payload(diagnosis_metrics)
    panel = _panel()

    update_model_combined_section(panel, payload)

    # The panel updated (no freeze): window label reflects the sparse
    # window and names the missing signal.
    assert (
        panel["win"].text
        == "5 aligned steps · representative r0 · partial: RESIDUAL"
    )
    # H2D is occurrence-driven and absent, so it's plain measured-zero.
    # residual_proxy is genuinely unmeasured (non-h2d), so it gets the
    # hatched sliver, not an empty segment indistinguishable from zero.
    # Measured phases scale against the iteration envelope (input_wait
    # 10 + step 100 = 110) minus the sliver's reserved width.
    keys = [key for _, key, _ in theme.PHASES]
    by_key = dict(zip(keys, panel["seg_divs"]))
    by_lab = dict(zip(keys, panel["seg_labs"]))
    assert by_key["h2d"].width == "0.000%"
    residual_seg = by_key["residual_proxy"]
    assert residual_seg.width == f"{_UNMEASURED_SLIVER_PCT:.1f}%"
    # The hatch is drawn in the PHASE'S OWN color (residual = gold), so the
    # sliver still identifies which phase, matching the legend dot.
    residual_color = dict((k, c) for _, k, c in theme.PHASES)["residual_proxy"]
    assert residual_seg.hatched
    assert residual_color in residual_seg.background
    # The sliver carries NO on-segment text -- it would overlap the hatch
    # illegibly; the missing phases are named in the window meta instead.
    assert by_lab["residual_proxy"].text == ""
    assert by_key["input_wait"].width == "8.545%"
    # An underivable residual shows n/a, never a fake 0%.
    assert "n/a" in panel["kpis"]["residual"].content
    assert panel["kpis"]["median"].content.startswith("100")


def test_step_time_dashboard_hero_measured_zero_is_not_partial() -> None:
    diagnosis_metrics = [
        _metric("input_wait", 10.0),
        _metric("h2d", 0.0),
        _metric("forward", 20.0),
        _metric("backward", 30.0),
        _metric("optimizer_step", 20.0),
        _metric("residual_proxy", 30.0),
        _metric("step_time", 100.0),
    ]
    payload = _payload(diagnosis_metrics)
    panel = _panel()

    update_model_combined_section(panel, payload)

    # A measured-zero H2D is complete coverage, not partial signals.
    assert panel["win"].text == "5 aligned steps · representative r0"
    assert "n/a" not in panel["kpis"]["residual"].content


def test_step_time_dashboard_hero_absent_h2d_is_not_partial() -> None:
    # H2D events are occurrence-driven: a run with no host-to-device
    # copies emits none, which is complete coverage, not partial.
    diagnosis_metrics = [
        _metric("input_wait", 10.0),
        _metric("forward", 20.0),
        _metric("backward", 30.0),
        _metric("optimizer_step", 20.0),
        _metric("residual_proxy", 30.0),
        _metric("step_time", 100.0),
    ]
    payload = _payload(diagnosis_metrics)
    panel = _panel()

    update_model_combined_section(panel, payload)

    assert panel["win"].text == "5 aligned steps · representative r0"
    keys = [key for _, key, _ in theme.PHASES]
    by_key = dict(zip(keys, panel["seg_divs"]))
    assert by_key["h2d"].width == "0.000%"


def test_step_time_dashboard_hero_step_time_only_extreme() -> None:
    payload = _payload([_metric("step_time", 100.0)])
    panel = _panel()

    update_model_combined_section(panel, payload)

    # Only h2d is exempt. Occurrence-driven governs INTERMITTENT presence
    # (seen once, gaps zero-filled); a phase never seen at all is
    # unavailable even when occurrence-driven, which is what
    # `_rank_metric_availability` does by only considering metrics it
    # actually observed. So a wholly absent optimizer_step IS partial.
    assert (
        panel["win"].text == "5 aligned steps · representative r0 · partial: "
        "IW,FWD,BWD,OPT,RESIDUAL"
    )
    keys = [key for _, key, _ in theme.PHASES]
    by_key = dict(zip(keys, panel["seg_divs"]))
    assert by_key["h2d"].width == "0.000%"
    assert not by_key["h2d"].hatched
    for key in (
        "input_wait",
        "forward",
        "backward",
        "optimizer_step",
        "residual_proxy",
    ):
        assert by_key[key].width == f"{_UNMEASURED_SLIVER_PCT:.1f}%"
        assert by_key[key].hatched
    assert "n/a" in panel["kpis"]["residual"].content
    assert panel["kpis"]["median"].content.startswith("100")


def test_absent_optimizer_step_is_not_a_measured_zero() -> None:
    # Narrow guard for the predicate itself. optimizer_step is
    # occurrence-driven, but "never observed in the whole window" is
    # absence, not an occurrence of zero: the canonical availability rule
    # drops a never-seen optimizer entirely, so the card must not render
    # it as a confident 0%.
    payload = _payload(
        [
            _metric("input_wait", 10.0),
            _metric("h2d", 10.0),
            _metric("forward", 20.0),
            _metric("backward", 30.0),
            _metric("residual_proxy", 20.0),
            _metric("step_time", 100.0),
        ]
    )
    panel = _panel()
    update_model_combined_section(panel, payload)

    keys = [key for _, key, _ in theme.PHASES]
    by_key = dict(zip(keys, panel["seg_divs"]))
    assert by_key["optimizer_step"].hatched
    assert (
        panel["win"].text
        == "5 aligned steps · representative r0 · partial: OPT"
    )


def test_residual_share_is_na_when_input_wait_is_unavailable() -> None:
    # residual_proxy derives from step_time and the compute phases, so it
    # survives an unmeasured input_wait -- but the DENOMINATOR does not.
    # The envelope substitutes zero for the missing wait, so any share
    # computed against it is a confident percentage of an unknown total.
    payload = _payload(
        [
            _metric("h2d", 10.0),
            _metric("forward", 20.0),
            _metric("backward", 30.0),
            _metric("optimizer_step", 20.0),
            _metric("residual_proxy", 10.0),
            _metric("step_time", 100.0),
        ]
    )
    panel = _panel()
    update_model_combined_section(panel, payload)

    assert "n/a" in panel["kpis"]["residual"].content

    # Control twin: with input_wait measured the denominator is whole, so
    # the same residual reports a real share.
    assert payload.window is not None
    with_wait = _payload(
        list(payload.window.metrics) + [_metric("input_wait", 10.0)]
    )
    control = _panel()
    update_model_combined_section(control, with_wait)
    assert "n/a" not in control["kpis"]["residual"].content


def test_partial_rank_coverage_includes_step_time() -> None:
    # step_time is not a ribbon phase, but it is the envelope every share
    # is computed against, so a rank missing it makes the window
    # incomplete exactly as the engine reports it.
    payload = _payload(
        [
            _metric("input_wait", 10.0),
            _metric("h2d", 10.0),
            _metric("forward", 20.0),
            _metric("backward", 30.0),
            _metric("optimizer_step", 20.0),
            _metric("residual_proxy", 20.0),
            _metric("step_time", 100.0),
        ],
        per_rank_timing={
            0: {
                "input_wait": 10.0,
                "h2d": 10.0,
                "forward": 20.0,
                "backward": 30.0,
                "optimizer_step": 20.0,
                "residual_proxy": 20.0,
                "step_time": 100.0,
                "total_step": 110.0,
            },
            1: {
                "input_wait": 10.0,
                "h2d": 10.0,
                "forward": 20.0,
                "backward": 30.0,
                "optimizer_step": 20.0,
                "total_step": 110.0,
            },
        },
    )
    panel = _panel()
    update_model_combined_section(panel, payload)

    assert (
        panel["win"].text
        == "5 aligned steps · representative r0 · partial: RESIDUAL,STEP"
    )


def test_step_time_dashboard_hero_dark_input_wait_is_not_measured_zero() -> (
    None
):
    # #88/#135 shape: the dataloader stream goes dark while everything
    # else keeps reporting. residual_proxy is a remainder (step_time -
    # h2d - forward - backward - optimizer_step) so it stays derivable
    # even though input_wait itself is unmeasured -- the ribbon must not
    # let the OTHER phases silently fill to 100% and read as "confirmed
    # no wait", which is exactly the class of bug this test pins.
    dark_input_wait = _payload(
        [
            _metric("h2d", 10.0),
            _metric("forward", 20.0),
            _metric("backward", 30.0),
            _metric("optimizer_step", 20.0),
            _metric("residual_proxy", 20.0),
            _metric("step_time", 100.0),
        ]
    )
    panel = _panel()
    update_model_combined_section(panel, dark_input_wait)

    keys = [key for _, key, _ in theme.PHASES]
    by_key = dict(zip(keys, panel["seg_divs"]))
    # Unmeasured, not a confirmed-zero: the hatched sliver, not 0%.
    assert by_key["input_wait"].width == f"{_UNMEASURED_SLIVER_PCT:.1f}%"
    assert by_key["input_wait"].hatched
    assert (
        panel["win"].text
        == "5 aligned steps · representative r0 · partial: IW"
    )

    # Control twin: input_wait genuinely measured as 0ms must render as
    # a real 0%-width segment, distinct from the unmeasured case above.
    measured_zero_input_wait = _payload(
        [
            _metric("input_wait", 0.0),
            _metric("h2d", 10.0),
            _metric("forward", 20.0),
            _metric("backward", 30.0),
            _metric("optimizer_step", 20.0),
            _metric("residual_proxy", 20.0),
            _metric("step_time", 100.0),
        ]
    )
    control_panel = _panel()
    update_model_combined_section(control_panel, measured_zero_input_wait)
    control_by_key = dict(zip(keys, control_panel["seg_divs"]))
    assert control_by_key["input_wait"].width == "0.000%"
    assert control_panel["win"].text == "5 aligned steps · representative r0"


def test_step_time_dashboard_hero_dead_run_shows_expired() -> None:
    # The CLI sibling distinguishes "no data yet" from "had data, now
    # expired" via StepCombinedComputer.had_ok; the dashboard hero must
    # do the same instead of freezing on the last complete view forever.
    good = _payload(
        [
            _metric("input_wait", 10.0),
            _metric("h2d", 10.0),
            _metric("forward", 20.0),
            _metric("backward", 30.0),
            _metric("optimizer_step", 20.0),
            _metric("residual_proxy", 20.0),
            _metric("step_time", 100.0),
        ],
        had_ok=True,
    )
    panel = _panel()
    update_model_combined_section(panel, good)
    assert panel["win"].text == "5 aligned steps · representative r0"
    assert panel["kpis"]["median"].content.startswith("100")

    # A never-had-data payload (cold start) must NOT clear the panel --
    # there is nothing to clear yet.
    cold_start = StepTimeResult(had_ok=False)
    cold_panel = _panel()
    update_model_combined_section(cold_panel, cold_start)
    assert cold_panel["win"].text == ""

    # TTL expired: the computer reports had_ok=True with empty metrics.
    # The dashboard must visibly clear rather than keep showing "good".
    expired = StepTimeResult(had_ok=True)
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
    phase was unmeasured outlives a later width-only update. A double
    that only appended style strings could never express that, which is
    how the stale-hatch bug reached review: the test could not fail.
    """
    from nicegui import ui

    real = ui.element("div")
    fake = _FakeSegment()
    for call in (
        "background:#1976d2; width:0%;",
        "width:6.0%; background:repeating-linear-gradient(45deg, "
        "#1976d2 0 2px, transparent 2px 5px);",
        "width:33.000%; background:#1976d2;",
    ):
        real.style(call)
        fake.style(call)

    assert dict(real._style) == fake.declarations
    # And the property the production fix depends on: restating the
    # background is what clears a previously-hatched segment.
    assert not fake.hatched


def test_step_time_dashboard_hero_recovered_phase_drops_the_hatch() -> None:
    # Stateful sparse -> complete. NiceGUI merges styles, so a phase that
    # was hatched while unmeasured must have its solid background
    # restored on recovery; otherwise it stays visually "unmeasured"
    # forever while reporting a real width.
    sparse = _payload(
        [
            _metric("input_wait", 10.0),
            _metric("h2d", 10.0),
            _metric("backward", 30.0),
            _metric("optimizer_step", 20.0),
            _metric("residual_proxy", 20.0),
            _metric("step_time", 100.0),
        ]
    )
    panel = _panel()
    update_model_combined_section(panel, sparse)

    keys = [key for _, key, _ in theme.PHASES]
    by_key = dict(zip(keys, panel["seg_divs"]))
    assert by_key["forward"].hatched
    assert (
        panel["win"].text
        == "5 aligned steps · representative r0 · partial: FWD"
    )

    # Forward starts reporting again.
    complete = _payload(
        [
            _metric("input_wait", 10.0),
            _metric("h2d", 10.0),
            _metric("forward", 20.0),
            _metric("backward", 30.0),
            _metric("optimizer_step", 20.0),
            _metric("residual_proxy", 20.0),
            _metric("step_time", 100.0),
        ]
    )
    update_model_combined_section(panel, complete)

    forward_color = dict((k, c) for _, k, c in theme.PHASES)["forward"]
    assert not by_key["forward"].hatched
    assert by_key["forward"].background == forward_color
    assert by_key["forward"].width != f"{_UNMEASURED_SLIVER_PCT:.1f}%"
    assert panel["win"].text == "5 aligned steps · representative r0"


def test_step_time_dashboard_hero_flags_partial_rank_coverage() -> None:
    # Aggregate presence is not coverage. Backward is measured on rank 0
    # and absent on rank 1, so the canonical diagnosis calls the window
    # INCOMPLETE; the card must not report it as fully covered just
    # because the aggregate metric exists.
    payload = _payload(
        [
            _metric("input_wait", 10.0),
            _metric("h2d", 10.0),
            _metric("forward", 20.0),
            _metric("backward", 30.0),
            _metric("optimizer_step", 20.0),
            _metric("residual_proxy", 20.0),
            _metric("step_time", 100.0),
        ],
        per_rank_timing={
            0: {
                "input_wait": 10.0,
                "h2d": 10.0,
                "forward": 20.0,
                "backward": 30.0,
                "optimizer_step": 20.0,
                "step_time": 100.0,
                "total_step": 110.0,
            },
            1: {
                "input_wait": 10.0,
                "h2d": 10.0,
                "forward": 20.0,
                "optimizer_step": 20.0,
                "step_time": 100.0,
                "total_step": 110.0,
            },
        },
    )
    panel = _panel()
    update_model_combined_section(panel, payload)

    assert (
        panel["win"].text
        == "5 aligned steps · representative r0 · partial: BWD,RESIDUAL"
    )

    # Control twin: every rank measured everything -> not partial.
    assert payload.window is not None
    symmetric = _payload(
        payload.window.metrics,
        per_rank_timing={
            rank: {
                "input_wait": 10.0,
                "h2d": 10.0,
                "forward": 20.0,
                "backward": 30.0,
                "optimizer_step": 20.0,
                "residual_proxy": 20.0,
                "step_time": 100.0,
                "total_step": 110.0,
            }
            for rank in (0, 1)
        },
    )
    control = _panel()
    update_model_combined_section(control, symmetric)
    assert control["win"].text == "5 aligned steps · representative r0"


def test_step_time_dashboard_hero_clears_when_step_envelope_missing() -> None:
    # Stateful: a COMPLETE window is drawn first, then a window arrives
    # with no step_time. Without the envelope there is no denominator, so
    # every share would be invented -- the card must clear rather than
    # leave the previous ribbon and KPIs standing as if they were current.
    complete = _payload(
        [
            _metric("input_wait", 10.0),
            _metric("h2d", 10.0),
            _metric("forward", 20.0),
            _metric("backward", 30.0),
            _metric("optimizer_step", 20.0),
            _metric("residual_proxy", 20.0),
            _metric("step_time", 100.0),
        ]
    )
    panel = _panel()
    update_model_combined_section(panel, complete)
    assert panel["win"].text == "5 aligned steps · representative r0"
    assert panel["kpis"]["median"].content.startswith("100")

    no_envelope = _payload(
        [
            _metric("input_wait", 10.0),
            _metric("forward", 20.0),
        ]
    )
    update_model_combined_section(panel, no_envelope)

    assert panel["win"].text == "step envelope unavailable"
    assert all(seg.width == "0%" for seg in panel["seg_divs"])
    assert all(
        kpi.content == theme.kval("—") for kpi in panel["kpis"].values()
    )


def test_step_time_dashboard_hero_uses_diagnosis_metrics() -> None:
    assert theme.PHASES[0][:2] == ("IW", "input_wait")

    # Self-consistent window shape: phases sum to input_wait + step.
    diagnosis_metrics = [
        _metric("input_wait", 10.0),
        _metric("h2d", 10.0),
        _metric("forward", 20.0),
        _metric("backward", 30.0),
        _metric("optimizer_step", 20.0),
        _metric("residual_proxy", 20.0),
        _metric("step_time", 100.0, worst=200.0),
    ]
    payload = _payload(diagnosis_metrics, clock="gpu")
    panel = _panel()

    update_model_combined_section(panel, payload)

    assert panel["seg_labs"][0].text == "IW"
    # input_wait 10 of the 110 ms iteration envelope.
    assert panel["seg_divs"][0].width == "9.091%"
    assert panel["win"].text == "5 aligned steps · representative r0"
    assert panel["kpis"]["median"].content.startswith("100")
    assert panel["kpis"]["worst"].content.startswith("200")
    assert not panel["kpis"]["median"].content.startswith("20")


def test_ribbon_selects_one_real_representative_rank() -> None:
    metrics = [
        _metric("input_wait", 50.0),
        _metric("forward", 45.0),
        _metric("backward", 15.0),
        _metric("optimizer_step", 7.5),
        _metric("residual_proxy", 32.5),
        _metric("step_time", 100.0),
    ]
    payload = _payload(
        metrics,
        per_rank_timing={
            0: {
                "input_wait": 0.0,
                "forward": 10.0,
                "backward": 20.0,
                "optimizer_step": 10.0,
                "residual_proxy": 60.0,
                "step_time": 100.0,
                "total_step": 100.0,
            },
            1: {
                "input_wait": 100.0,
                "forward": 80.0,
                "backward": 10.0,
                "optimizer_step": 5.0,
                "residual_proxy": 5.0,
                "step_time": 100.0,
                "total_step": 200.0,
            },
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
        [
            _metric("input_wait", 10.0),
            _metric("forward", 40.0),
            _metric("backward", 40.0),
            _metric("optimizer_step", 10.0),
            _metric("step_time", 100.0),
        ],
        per_rank_timing={
            0: {
                "input_wait": 10.0,
                "forward": 40.0,
                "step_time": 100.0,
                "total_step": 110.0,
            },
            1: {
                "backward": 40.0,
                "optimizer_step": 10.0,
                "step_time": 120.0,
            },
        },
    )
    panel = _panel()

    update_model_combined_section(panel, payload)

    assert "no coherent phase cohort" in panel["win"].text
    assert all(segment.width == "0%" for segment in panel["seg_divs"])
    assert panel["kpis"]["median"].content.startswith("100")
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

    result = StepCombinedComputer(str(db_path), window_size=30).compute_cli()
    assert result.window is not None
    assert result.window.per_rank_timing[0]["residual_proxy"] == pytest.approx(
        10.0
    )
    diagnosis = diagnose_step_time_window(
        result.window,
        policy=LIVE_STEP_TIME_POLICY,
    )
    assert diagnosis.metric_attribution["residual_proxy"][
        "share_pct"
    ] == pytest.approx(0.05)

    summary = load_step_time_section_data(str(db_path), max_rows=30)
    assert summary.per_global_rank_summary[0].avg_residual_ms == pytest.approx(
        10.0
    )
    assert summary.per_global_rank_summary[0].avg_h2d_ms is None

    renderer = StepCombinedRenderer(str(db_path))
    monkeypatch.setattr(renderer, "_payload", lambda: result)
    console = Console(record=True, width=140, color_system=None)
    console.print(renderer.get_panel_renderable())
    assert "5.0%" in console.export_text()

    diagnostics = build_model_diagnostics_payload(
        step_time_window=result.window,
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
