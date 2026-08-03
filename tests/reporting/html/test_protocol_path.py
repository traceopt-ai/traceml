from pathlib import Path

from traceml_ai.sdk.protocol import (
    get_final_summary_html_path,
    get_final_summary_json_path,
)


def test_html_path_is_sibling_of_json(tmp_path) -> None:
    html = get_final_summary_html_path(tmp_path)
    assert html == (tmp_path / "final_summary.html").resolve()
    # Sits next to the JSON artifact in the same session root.
    assert html.parent == get_final_summary_json_path(tmp_path).parent


def test_html_path_accepts_str(tmp_path) -> None:
    assert (
        get_final_summary_html_path(str(tmp_path))
        == (Path(tmp_path) / "final_summary.html").resolve()
    )
