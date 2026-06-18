import traceml_ai.reporting.html as html_pkg
from traceml_ai.reporting.final import write_summary_artifacts


def _write(payload, tmp_path, *, write_html):
    session_root = tmp_path / "sess"
    db_path = tmp_path / "agg" / "telemetry"
    write_summary_artifacts(
        db_path=str(db_path),
        payload=payload,
        session_root=str(session_root),
        write_html=write_html,
    )
    return session_root


def test_write_html_true_emits_html_next_to_json(
    make_payload, tmp_path
) -> None:
    session_root = _write(make_payload(), tmp_path, write_html=True)
    html = session_root / "final_summary.html"
    assert (session_root / "final_summary.json").exists()
    assert html.exists()
    assert html.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_write_html_false_emits_no_html(make_payload, tmp_path) -> None:
    session_root = _write(make_payload(), tmp_path, write_html=False)
    assert (session_root / "final_summary.json").exists()
    assert not (session_root / "final_summary.html").exists()


def test_html_failure_never_blocks_json_txt(
    make_payload, tmp_path, monkeypatch, capsys
) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("render exploded")

    monkeypatch.setattr(html_pkg, "write_html_report", _boom)

    # Must not raise, and JSON/TXT must still be on disk (best-effort C1).
    session_root = _write(make_payload(), tmp_path, write_html=True)

    assert (session_root / "final_summary.json").exists()
    assert (session_root / "final_summary.txt").exists()
    assert not (session_root / "final_summary.html").exists()
    assert "[TraceML]" in capsys.readouterr().err
