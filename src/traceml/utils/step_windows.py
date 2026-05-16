# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Small helpers for step-indexed telemetry windows."""

from __future__ import annotations

from typing import Mapping


def common_suffix_steps(
    per_rank_steps: Mapping[int, Mapping[int, object]],
    max_rows: int,
) -> list[int]:
    """Return the latest step ids present for every observed rank."""
    if not per_rank_steps:
        return []

    step_sets: list[set[int]] = []
    for step_map in per_rank_steps.values():
        if not step_map:
            return []
        step_sets.append(set(int(step) for step in step_map.keys()))

    common = set.intersection(*step_sets) if step_sets else set()
    if not common:
        return []

    steps = sorted(common)
    return steps[-max(1, int(max_rows)) :]


__all__ = ["common_suffix_steps"]
