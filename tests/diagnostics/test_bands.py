from traceml.diagnostics.bands import BandThresholds, format_band_value


def test_band_thresholds_classify_low_normal_high_and_very_high() -> None:
    thresholds = BandThresholds(
        low_below=30.0,
        high_at=80.0,
        very_high_at=90.0,
    )

    assert thresholds.classify(None) is None
    assert thresholds.classify(20.0) == "low"
    assert thresholds.classify(50.0) == "normal"
    assert thresholds.classify(80.0) == "high"
    assert thresholds.classify(90.0) == "very_high"


def test_format_band_value_is_compact_and_null_safe() -> None:
    assert format_band_value("CPU", "normal", 42.2) == "CPU normal 42%"
    assert format_band_value("GPU memory", None, 42.2) is None
    assert format_band_value("GPU memory", "high", None) is None
