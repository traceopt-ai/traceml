"""Header coverage chips mirror the terminal card's coverage segments."""

import re

from traceml_ai.reporting.html import render_html_report


def _chips(html: str) -> str:
    match = re.search(r'<div class="meta-chips">(.*?)</div>', html)
    assert match is not None
    return match.group(1)


def _coverage_payload(
    make_payload,
    *,
    world_size,
    ranks_seen,
    ranks_used,
    nodes_expected,
    nodes_observed,
    nodes_partial,
    steps,
    alignment,
):
    payload = make_payload(
        meta={
            "run_name": "coverage-run",
            "mode": "multi_node",
            "world_size": world_size,
            "nodes_observed": nodes_observed,
            "gpus_observed": world_size,
        }
    )
    payload["step_time"]["metadata"].update(
        {
            "global_ranks_seen": ranks_seen,
            "global_ranks_used": ranks_used,
        }
    )
    payload["step_time"]["global"]["window"] = {
        "kind": "aligned",
        "alignment": alignment,
        "steps_analyzed": steps,
    }
    payload["system"]["metadata"].update(
        {
            "nodes_expected": nodes_expected,
            "nodes_observed": nodes_observed,
            "nodes_coverage": f"{nodes_observed}/{nodes_expected}",
            "nodes_partial": nodes_partial,
        }
    )
    return payload


def test_partial_run_header_shows_node_rank_and_step_shortfall(
    make_payload,
) -> None:
    payload = _coverage_payload(
        make_payload,
        world_size=4,
        ranks_seen=4,
        ranks_used=3,
        nodes_expected=4,
        nodes_observed=2,
        nodes_partial=True,
        steps=40,
        alignment="common_steps",
    )
    chips = _chips(render_html_report(payload))
    assert '<span class="chip"><b>3/4 ranks</b></span>' in chips
    assert '<span class="chip"><b>2/4 nodes</b></span>' in chips
    assert '<span class="chip"><b>40 common steps</b></span>' in chips
    # The coverage chips replace the bare counts they already carry.
    assert "world size" not in chips
    assert "nodes <b>2</b>" not in chips


def test_complete_run_header_states_full_coverage_without_caveat(
    make_payload,
) -> None:
    payload = _coverage_payload(
        make_payload,
        world_size=4,
        ranks_seen=4,
        ranks_used=4,
        nodes_expected=4,
        nodes_observed=4,
        nodes_partial=False,
        steps=40,
        alignment="step",
    )
    chips = _chips(render_html_report(payload))
    assert "<b>4/4 ranks</b>" in chips
    assert "<b>4/4 nodes</b>" in chips
    assert "<b>40 steps analyzed</b>" in chips
    for shortfall in ("1/4", "2/4", "3/4", "partial"):
        assert shortfall not in chips


def test_absent_coverage_fields_render_no_chip_and_no_zero(
    make_payload, make_section
) -> None:
    payload = make_payload(
        meta={"run_name": "bare"},
        step_time=make_section(
            metric_names=["step_time_ms"],
            average={"step_time_ms": 100.0},
            rows={},
        ),
        system=make_section(
            metric_names=["gpu_util_percent"],
            average={"gpu_util_percent": 50.0},
            by="node_rank",
            rows={},
        ),
    )
    chips = _chips(render_html_report(payload))
    for word in ("rank", "node", "step", "world size"):
        assert word not in chips
    assert "<b>0" not in chips


def test_header_chips_match_terminal_card_coverage(make_payload) -> None:
    from traceml_ai.reporting.terminal_card.card import (
        summary_header_coverage,
    )

    payload = _coverage_payload(
        make_payload,
        world_size=4,
        ranks_seen=4,
        ranks_used=3,
        nodes_expected=4,
        nodes_observed=2,
        nodes_partial=True,
        steps=40,
        alignment="common_steps",
    )
    coverage = summary_header_coverage(payload)
    chips = _chips(render_html_report(payload))
    assert (coverage.ranks, coverage.nodes, coverage.steps) == (
        "3/4 ranks",
        "2/4 nodes",
        "40 common steps",
    )
    for text in (coverage.ranks, coverage.nodes, coverage.steps):
        assert f"<b>{text}</b>" in chips


def test_watch_payload_header_uses_watch_card_routing(make_payload) -> None:
    payload = _coverage_payload(
        make_payload,
        world_size=4,
        ranks_seen=4,
        ranks_used=3,
        nodes_expected=4,
        nodes_observed=2,
        nodes_partial=True,
        steps=40,
        alignment="common_steps",
    )
    payload["text"] = "TraceML Watch Summary\n"
    payload["process"]["metadata"]["global_ranks_used"] = 4
    chips = _chips(render_html_report(payload))
    # Watch cards take ranks from Process and never state Step Time steps.
    assert "<b>4/4 ranks</b>" in chips
    assert "<b>2/4 nodes</b>" in chips
    assert "steps" not in chips
