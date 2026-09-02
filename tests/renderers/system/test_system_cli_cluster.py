# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""The multi-node terminal view, over a node that did not report.

Every metric in this builder already counts its own readings and abstains
when it has none: memory, temperature, headroom and the GPU mean all pass
through ``_optional_float`` and a per-metric list. Host CPU was the one
that did not, and it was also the one field of ``_NodeSystemSample`` that
was not optional.
"""

from __future__ import annotations

import sqlite3

from traceml_ai.renderers.system.cli_compute import SystemCLIComputer


def _null_cpu_on(path: str, hostname: str) -> None:
    """Drop one node's CPU reading, leaving the rest of the run intact."""
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "UPDATE system_samples SET cpu_percent = NULL WHERE hostname = ?",
            (hostname,),
        )
        conn.commit()
    finally:
        conn.close()


def test_a_node_that_did_not_report_cpu_is_absent_from_the_rollup(
    system_db,
):
    """One silent node must not pull the cluster median toward zero.

    The median is over nodes, so a fabricated 0.0 does not just mislead
    about that node: it moves the number the whole cluster is judged by,
    and it wins the "worst node" comparison outright because zero is the
    lowest CPU any node can report.
    """
    path = system_db(ticks=5, cpu=lambda seq: 80.0, hostnames=("a", "b"))
    _null_cpu_on(path, "b")

    out = SystemCLIComputer(path).compute()

    assert out["view"] == "cluster"
    cpu = out["metrics"]["cpu"]
    # Node a reported 80 and node b reported nothing, so the median is
    # node a's reading rather than the midpoint of 80 and a phantom 0.
    assert cpu["median"] == 80.0
    assert cpu["worst"] == 80.0
    assert cpu["worst_node"] != "b"


def test_a_cluster_with_no_cpu_readings_abstains(system_db):
    """No node reported, so there is no median to state."""
    path = system_db(ticks=5, cpu=lambda seq: 80.0, hostnames=("a", "b"))
    _null_cpu_on(path, "a")
    _null_cpu_on(path, "b")

    out = SystemCLIComputer(path).compute()

    assert out["view"] == "cluster"
    assert out["metrics"]["cpu"]["median"] is None
    assert out["metrics"]["cpu"]["worst"] is None
