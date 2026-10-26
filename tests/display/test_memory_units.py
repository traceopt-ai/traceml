# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Every surface renders a byte count in binary units, labelled as such."""

from __future__ import annotations

import pytest

from traceml_ai.aggregator.display_drivers.nicegui_sections import (
    formatting as dashboard,
)
from traceml_ai.aggregator.display_drivers.nicegui_sections import (
    step_memory_section,
    system_section,
)
from traceml_ai.measurements.system import (
    PerGPUSummary,
    SystemNodeSummary,
    node_gpu_headroom_min_bytes,
)
from traceml_ai.reporting.html.textutils import fmt_value
from traceml_ai.reporting.summaries.summary_formatting import bytes_to_gb
from traceml_ai.reporting.terminal_card import common as card
from traceml_ai.reporting.terminal_card.evidence import _bytes
from traceml_ai.utils.formatting import fmt_mem_new, format_memory

GIB = 1024**3
MIB = 1024**2
CAPACITY = 16 * GIB  # a 16 GiB device: 17.18 decimal GB


def test_dashboard_cards_agree_on_one_capacity() -> None:
    expected = ("8.0", "/ 16.0 GiB")
    assert dashboard.format_gb_pair(8 * GIB, CAPACITY) == expected
    assert system_section.format_gb_pair(8 * GIB, CAPACITY) == expected


def test_dashboard_step_memory_uses_gibibytes() -> None:
    assert step_memory_section.BYTES_PER_GB == GIB


def test_summary_card_and_html_use_gibibytes() -> None:
    assert bytes_to_gb(CAPACITY) == 16.0
    assert card.format_capacity(CAPACITY) == "16.0 GiB"
    assert card.format_capacity(5 * MIB) == "5.0 MiB"
    assert _bytes(CAPACITY) == "+16.0 GiB"
    assert _bytes(5 * MIB) == "+5.0 MiB"
    assert fmt_value("peak_bytes", CAPACITY) == "16.00&thinsp;GiB"
    assert fmt_value("peak_bytes", 5 * MIB) == "5&thinsp;MiB"


def test_cli_formatters_label_binary_units() -> None:
    assert fmt_mem_new(CAPACITY) == "16.0 GiB"
    assert fmt_mem_new(512 * MIB) == "512 MiB"
    assert format_memory(CAPACITY) == (16.0, "GiB")


def test_gpu_headroom_stays_in_bytes() -> None:
    gpu = PerGPUSummary(
        gpu_idx=0,
        mem_total_bytes=float(CAPACITY),
        mem_peak_bytes=float(6 * GIB),
    )
    node = SystemNodeSummary.__new__(SystemNodeSummary)
    object.__setattr__(node, "per_gpu", {0: gpu})

    assert node_gpu_headroom_min_bytes(node) == pytest.approx(10 * GIB)


def test_card_switches_units_at_one_point() -> None:
    # 0.1 GiB is 107,374,182.4 bytes; both card paths switch there.
    below, above = 107_374_182, 107_374_183
    assert card.format_capacity(below).endswith(" MiB")
    assert _bytes(below).endswith(" MiB")
    assert card.format_capacity(above).endswith(" GiB")
    assert _bytes(above).endswith(" GiB")
