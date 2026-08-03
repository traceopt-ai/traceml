import argparse
import json

import pytest

from traceml_ai.launcher.cli import build_parser
from traceml_ai.launcher.commands import run_view, validate_launch_args


def _launch_args(**overrides) -> argparse.Namespace:
    base = dict(
        mode="summary",
        no_history=False,
        summary_window_rows=10000,
        trace_max_steps=None,
        html_report=True,
        nnodes=1,
        nproc_per_node=1,
        node_rank=0,
        master_addr="127.0.0.1",
        master_port=29500,
        run_name=None,
        session_id=None,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_run_parser_accepts_html_report_flag() -> None:
    args = build_parser().parse_args(["run", "train.py", "--html-report"])
    assert args.html_report is True


def test_html_report_defaults_off() -> None:
    args = build_parser().parse_args(["run", "train.py"])
    assert args.html_report is False


def test_watch_parser_accepts_html_report_flag() -> None:
    args = build_parser().parse_args(["watch", "train.py", "--html-report"])
    assert args.html_report is True


def test_view_html_flag_optional_value() -> None:
    p = build_parser()
    assert p.parse_args(["view", "s.json"]).html is None
    assert p.parse_args(["view", "s.json", "--html"]).html == ""
    assert p.parse_args(["view", "s.json", "--html", "o.html"]).html == (
        "o.html"
    )


def test_validate_rejects_html_report_with_no_history() -> None:
    with pytest.raises(SystemExit, match="--html-report requires history"):
        validate_launch_args(_launch_args(mode="cli", no_history=True))


def test_validate_allows_html_report_with_history() -> None:
    validate_launch_args(_launch_args(mode="cli", no_history=False))


def test_run_view_html_writes_report(tmp_path, make_payload, capsys) -> None:
    json_path = tmp_path / "final_summary.json"
    json_path.write_text(json.dumps(make_payload()), encoding="utf-8")
    args = argparse.Namespace(summary=str(json_path), html="")

    run_view(args)

    out_html = tmp_path / "final_summary.html"
    assert out_html.exists()
    assert str(out_html) in capsys.readouterr().out


def test_run_view_without_html_prints_text(
    tmp_path, make_payload, capsys
) -> None:
    json_path = tmp_path / "final_summary.json"
    json_path.write_text(json.dumps(make_payload()), encoding="utf-8")
    args = argparse.Namespace(summary=str(json_path), html=None)

    run_view(args)

    out = capsys.readouterr().out
    assert "TraceML summary card" in out
    assert not (tmp_path / "final_summary.html").exists()
