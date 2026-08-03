from traceml_ai.reporting.html.textutils import esc


def test_esc_escapes_html_metacharacters() -> None:
    assert (
        esc("<script>alert('x')</script>")
        == "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;"
    )


def test_esc_escapes_ampersand_and_double_quotes() -> None:
    assert esc('a & b "c"') == "a &amp; b &quot;c&quot;"


def test_esc_strips_c0_control_chars_including_ansi_escape() -> None:
    # ANSI ESC (0x1b) and NUL (0x00) must not survive into the HTML.
    assert esc("ok\x1b[31mred\x1b[0m\x00") == "ok[31mred[0m"


def test_esc_keeps_tab_and_newline() -> None:
    assert esc("a\tb\nc") == "a\tb\nc"


def test_esc_renders_none_as_em_dash_placeholder() -> None:
    assert esc(None) == "—"


def test_esc_coerces_non_string_scalars() -> None:
    assert esc(4) == "4"
    assert esc(1.5) == "1.5"
