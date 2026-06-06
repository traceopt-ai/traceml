# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Node-aligned cluster rollups for the terminal System panel."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .common import SystemMetricsDB

CLI_CLUSTER_COVERAGE_RATIO = 0.90
CLI_CLUSTER_WINDOW_ROWS = 1_000


@dataclass(frozen=True)
class SystemCLIClusterSelection:
    """Selected aligned rows for one terminal cluster snapshot."""

    rows: List[Any]
    expected_nodes: int
    partial: bool


@dataclass(frozen=True)
class _NodeSystemSample:
    """Aligned system telemetry for one node."""

    label: str
    seq: int
    cpu_percent: float
    ram_percent: Optional[float]
    gpu_util_percent: Optional[float]
    gpu_mem_percent: Optional[float]
    gpu_temp_c: Optional[float]
    gpu_headroom_bytes: Optional[float]


@dataclass(frozen=True)
class _MetricRollup:
    """Median value and worst-node value for one terminal metric."""

    median: Optional[float]
    worst: Optional[float]
    worst_node: Optional[str]


class SystemCLIClusterBuilder:
    """
    Build node-aware terminal snapshots from projected SystemSampler rows.

    A cluster snapshot compares nodes only within one sampler sequence. The
    preferred sequence is the newest `seq` with at least 90% expected-node
    coverage. If metadata is inconsistent or heterogeneous, observed nodes are
    used as the denominator to avoid overstating coverage.
    """

    def __init__(self, db: SystemMetricsDB) -> None:
        self._db = db

    def select(self, rows: Sequence[Any]) -> SystemCLIClusterSelection:
        """Select the aligned sequence used for terminal comparison."""
        latest_by_seq_node: Dict[int, Dict[str, Any]] = {}
        observed_nodes = {
            self._node_key(row) for row in rows if row["seq"] is not None
        }
        expected_nodes = self._expected_nodes(
            rows, observed=len(observed_nodes)
        )

        for row in rows:
            seq = row["seq"]
            if seq is None:
                continue
            seq_id = int(seq)
            node_key = self._node_key(row)
            per_node = latest_by_seq_node.setdefault(seq_id, {})
            existing = per_node.get(node_key)
            if existing is None or int(row["id"] or 0) >= int(
                existing["id"] or 0
            ):
                per_node[node_key] = row

        if not latest_by_seq_node:
            return SystemCLIClusterSelection(
                rows=[rows[-1]],
                expected_nodes=max(1, expected_nodes),
                partial=False,
            )

        required = max(
            1,
            int(math.ceil(CLI_CLUSTER_COVERAGE_RATIO * expected_nodes)),
        )
        for seq_id in sorted(latest_by_seq_node.keys(), reverse=True):
            selected = list(latest_by_seq_node[seq_id].values())
            if len(selected) >= required:
                return SystemCLIClusterSelection(
                    rows=sorted(selected, key=self._node_sort_key),
                    expected_nodes=expected_nodes,
                    partial=False,
                )

        best_seq = max(
            latest_by_seq_node,
            key=lambda seq_id: (
                len(latest_by_seq_node[seq_id]),
                int(seq_id),
            ),
        )
        return SystemCLIClusterSelection(
            rows=sorted(
                latest_by_seq_node[best_seq].values(),
                key=self._node_sort_key,
            ),
            expected_nodes=expected_nodes,
            partial=True,
        )

    def build(
        self,
        conn: sqlite3.Connection,
        selection: SystemCLIClusterSelection,
    ) -> Optional[Dict[str, Any]]:
        """Build a terminal cluster payload for the selected rows."""
        rows = selection.rows
        if len(rows) <= 1:
            return None

        sample_keys = [
            (row["global_rank"], row["seq"])
            for row in rows
            if row["seq"] is not None
        ]
        gpu_rows = self._db.fetch_gpu_rows_for_samples(conn, sample_keys)
        gpu_by_key = self._db.group_gpu_rows_by_global_rank_seq(gpu_rows)

        node_samples: List[_NodeSystemSample] = []
        for row in rows:
            seq = row["seq"]
            if seq is None:
                continue
            key = (row["global_rank"], int(seq))
            node_samples.append(
                self._node_sample_from_row(
                    row,
                    gpu_rows=gpu_by_key.get(key, []),
                )
            )

        if len(node_samples) <= 1:
            return None

        node_count = len(node_samples)
        title_suffix = (
            "(med/worst, partial view, "
            f"nodes {node_count}/{selection.expected_nodes})"
            if selection.partial
            else f"(med/worst, nodes {node_count}/{selection.expected_nodes})"
        )

        return {
            "view": "cluster",
            "title_suffix": title_suffix,
            "seq": int(node_samples[0].seq),
            "node_count": node_count,
            "expected_nodes": selection.expected_nodes,
            "partial": selection.partial,
            "gpu_available": any(
                sample.gpu_util_percent is not None for sample in node_samples
            ),
            "metrics": {
                "cpu": self._rollup(
                    node_samples,
                    value=lambda sample: sample.cpu_percent,
                    higher_is_worse=True,
                ),
                "ram": self._rollup(
                    node_samples,
                    value=lambda sample: sample.ram_percent,
                    higher_is_worse=True,
                ),
                "gpu_util": self._rollup(
                    node_samples,
                    value=lambda sample: sample.gpu_util_percent,
                    higher_is_worse=False,
                ),
                "gpu_mem": self._rollup(
                    node_samples,
                    value=lambda sample: sample.gpu_mem_percent,
                    higher_is_worse=True,
                ),
                "gpu_temp": self._rollup(
                    node_samples,
                    value=lambda sample: sample.gpu_temp_c,
                    higher_is_worse=True,
                ),
                "gpu_headroom": self._rollup(
                    node_samples,
                    value=lambda sample: sample.gpu_headroom_bytes,
                    higher_is_worse=False,
                ),
            },
        }

    def _node_sample_from_row(
        self,
        row: Any,
        *,
        gpu_rows: Sequence[Any],
    ) -> _NodeSystemSample:
        """Convert one system row plus GPU rows into node-level metrics."""
        gpu_utils: List[float] = []
        gpu_mem_percents: List[float] = []
        gpu_temps: List[float] = []
        gpu_headrooms: List[float] = []

        for gpu in gpu_rows:
            util = self._optional_float(gpu["util"])
            mem_used = self._optional_float(gpu["mem_used_bytes"])
            mem_total = self._optional_float(gpu["mem_total_bytes"])
            temp = self._optional_float(gpu["temperature_c"])

            if util is not None:
                gpu_utils.append(util)
            if (
                mem_used is not None
                and mem_total is not None
                and mem_total > 0.0
            ):
                gpu_mem_percents.append(mem_used / mem_total * 100.0)
                gpu_headrooms.append(max(mem_total - mem_used, 0.0))
            if temp is not None:
                gpu_temps.append(temp)

        ram_used = self._optional_float(row["ram_used_bytes"])
        ram_total = self._optional_float(row["ram_total_bytes"])
        return _NodeSystemSample(
            label=self._node_label(row),
            seq=int(row["seq"]),
            cpu_percent=float(row["cpu_percent"] or 0.0),
            ram_percent=(
                ram_used / ram_total * 100.0
                if (
                    ram_used is not None
                    and ram_total is not None
                    and ram_total > 0.0
                )
                else None
            ),
            gpu_util_percent=(
                sum(gpu_utils) / float(len(gpu_utils)) if gpu_utils else None
            ),
            gpu_mem_percent=(
                max(gpu_mem_percents) if gpu_mem_percents else None
            ),
            gpu_temp_c=max(gpu_temps) if gpu_temps else None,
            gpu_headroom_bytes=min(gpu_headrooms) if gpu_headrooms else None,
        )

    def _expected_nodes(self, rows: Sequence[Any], *, observed: int) -> int:
        """Infer expected nodes without forcing homogeneous layouts."""
        candidates = set()
        for row in rows:
            world_size = self._optional_int(row["world_size"])
            local_world_size = self._optional_int(row["local_world_size"])
            if (
                world_size is not None
                and local_world_size is not None
                and world_size > 0
                and local_world_size > 0
            ):
                candidates.add(int(math.ceil(world_size / local_world_size)))
        if len(candidates) == 1:
            return max(1, candidates.pop())
        return max(1, int(observed))

    def _rollup(
        self,
        samples: Sequence[_NodeSystemSample],
        *,
        value: Callable[[_NodeSystemSample], Optional[float]],
        higher_is_worse: bool,
    ) -> Dict[str, Optional[float]]:
        """Return median/worst value and worst-node label."""
        pairs = [
            (float(metric), sample.label)
            for sample in samples
            if (metric := value(sample)) is not None
        ]
        if not pairs:
            return asdict(_MetricRollup(None, None, None))

        values = [metric for metric, _label in pairs]
        worst_value, worst_label = (
            max(pairs, key=lambda item: item[0])
            if higher_is_worse
            else min(pairs, key=lambda item: item[0])
        )
        return asdict(
            _MetricRollup(
                median=float(median(values)),
                worst=float(worst_value),
                worst_node=worst_label,
            )
        )

    def _node_key(self, row: Any) -> str:
        """Return a stable grouping key for one node."""
        node_rank = self._optional_int(row["node_rank"])
        if node_rank is not None:
            return f"node:{node_rank}"
        global_rank = self._optional_int(row["global_rank"])
        return (
            f"global:{global_rank}" if global_rank is not None else "unknown"
        )

    def _node_label(self, row: Any) -> str:
        """Return the compact terminal label for one node."""
        node_rank = self._optional_int(row["node_rank"])
        if node_rank is not None:
            return f"n{node_rank}"
        global_rank = self._optional_int(row["global_rank"])
        return f"g{global_rank}" if global_rank is not None else "n/a"

    def _node_sort_key(self, row: Any) -> Tuple[int, str]:
        """Sort node rows in a predictable order."""
        node_rank = self._optional_int(row["node_rank"])
        if node_rank is not None:
            return (0, f"{node_rank:09d}")
        global_rank = self._optional_int(row["global_rank"])
        return (1, f"{global_rank:09d}" if global_rank is not None else "")

    def _optional_float(self, value: Any) -> Optional[float]:
        """Best-effort float conversion for SQLite values."""
        try:
            return float(value) if value is not None else None
        except Exception:
            return None

    def _optional_int(self, value: Any) -> Optional[int]:
        """Best-effort integer conversion for SQLite values."""
        try:
            return int(value) if value is not None else None
        except Exception:
            return None


__all__ = [
    "CLI_CLUSTER_WINDOW_ROWS",
    "SystemCLIClusterBuilder",
    "SystemCLIClusterSelection",
]
