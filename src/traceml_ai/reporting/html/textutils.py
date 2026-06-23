# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Text helpers for the HTML report: safe escaping of payload strings."""

from __future__ import annotations

import html
from typing import Any

from ..summaries.summary_formatting import bytes_to_gb

# C0 control characters (0x00-0x1F) are stripped before escaping so that
# raw terminal/ANSI bytes from an untrusted payload never reach the HTML.
# Tab and newline are kept because they are meaningful whitespace.
_KEEP = {"\t", "\n"}
_PLACEHOLDER = "—"  # em dash, shown for missing values


def esc(value: Any) -> str:
    """
    Render any payload scalar as safe HTML display text.

    Coerces ``value`` to a string, strips C0 control characters (except tab
    and newline), then HTML-escapes ``< > & " '``. ``None`` renders as an
    em-dash placeholder. ``view --html`` accepts arbitrary JSON, so every
    payload-derived string must pass through here.
    """
    if value is None:
        return _PLACEHOLDER
    text = value if isinstance(value, str) else str(value)
    stripped = "".join(ch for ch in text if ch in _KEEP or ord(ch) >= 0x20)
    return html.escape(stripped, quote=True)


# Friendly display labels for known metric keys; unknown keys render raw.
_METRIC_LABELS = {
    "total_step_ms": "Total step",
    "dataloader_ms": "Dataloader",
    "h2d_ms": "H2D",
    "compute_ms": "Compute",
    "residual_ms": "Residual",
    "forward_ms": "Forward",
    "backward_ms": "Backward",
    "optimizer_ms": "Optimizer",
    "peak_allocated_bytes": "Peak allocated",
    "peak_reserved_bytes": "Peak reserved",
    "cpu_percent": "CPU",
    "cpu_capacity_percent": "CPU capacity",
    "ram_bytes": "RAM",
    "ram_percent": "RAM %",
    "gpu_util_percent": "GPU util",
    "gpu_mem_bytes": "GPU mem",
    "gpu_mem_percent": "GPU mem %",
    "gpu_mem_used_bytes": "GPU used",
    "gpu_mem_reserved_bytes": "GPU reserved",
    "gpu_mem_reserved_percent": "GPU reserved %",
    "gpu_mem_headroom_bytes": "Headroom",
    "gpu_headroom_bytes": "Headroom",
    "gpu_temp_c": "GPU temp",
    "gpu_power_w": "GPU power",
}


def metric_label(name: str) -> str:
    """Map a payload metric key to a friendly display label (raw fallback)."""
    return _METRIC_LABELS.get(name, name)


def fmt_value(name: str, value: Any) -> str:
    """
    Format a metric value, resolving the unit from the metric-name suffix.

    Parity with the text card: bytes use decimal GB (``bytes_to_gb``, 1e9)
    and milliseconds use one decimal. Non-numeric values are escaped as text.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return esc(value)
    if name.endswith("_ms"):
        return f"{value:,.1f}&thinsp;ms"
    if name.endswith("_bytes"):
        gb = bytes_to_gb(value) or 0.0
        if gb >= 1.0:
            return f"{gb:,.2f}&thinsp;GB"
        return f"{value / 1e6:,.0f}&thinsp;MB"
    if name.endswith("_percent"):
        return f"{value:,.1f}%"
    if name.endswith("_c"):
        return f"{value:,.0f}&thinsp;&deg;C"
    if name.endswith("_w"):
        return f"{value:,.0f}&thinsp;W"
    return f"{value:,.2f}"


__all__ = ["esc", "fmt_value", "metric_label"]
