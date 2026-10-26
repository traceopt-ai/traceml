# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Read the measurement families used by one recent final report."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Callable, Dict, Mapping, Optional, TypeVar

from traceml_ai.measurements.process import (
    ProcessMeasurements,
    read_process_measurements,
)
from traceml_ai.measurements.step_memory import (
    StepMemoryMeasurements,
    read_step_memory_measurements,
)
from traceml_ai.measurements.system import (
    SystemMeasurements,
    read_system_measurements,
)
from traceml_ai.measurements.window import AnalysisWindow
from traceml_ai.step_time.model import (
    StepTimeLoadRequest,
    StepTimeRepositorySnapshot,
)
from traceml_ai.step_time.sqlite import SQLiteStepTimeRepository

MeasurementT = TypeVar("MeasurementT")


@dataclass(frozen=True)
class RecentMeasurements:
    """Measurement results read through one caller-owned SQLite snapshot."""

    system: Optional[SystemMeasurements] = None
    process: Optional[ProcessMeasurements] = None
    step_time: Optional[StepTimeRepositorySnapshot] = None
    step_memory: Optional[StepMemoryMeasurements] = None
    errors: Mapping[str, Exception] = field(default_factory=dict)


def _read_family(
    name: str,
    reader: Callable[[], MeasurementT],
    errors: Dict[str, Exception],
) -> Optional[MeasurementT]:
    """Read one family without hiding successful sibling reads."""
    try:
        return reader()
    except Exception as exc:
        errors[name] = exc
        return None


def read_recent(
    connection: sqlite3.Connection,
    selection: AnalysisWindow,
) -> RecentMeasurements:
    """Read recent measurements without owning the surrounding transaction."""
    errors: Dict[str, Exception] = {}
    system = _read_family(
        "system",
        lambda: read_system_measurements(
            connection,
            analysis_window=selection,
        ),
        errors,
    )
    process = _read_family(
        "process",
        lambda: read_process_measurements(
            connection,
            analysis_window=selection,
        ),
        errors,
    )
    step_time = _read_family(
        "step_time",
        lambda: SQLiteStepTimeRepository(connection).load_summary(
            StepTimeLoadRequest(
                start_step=selection.start_step,
                end_step=selection.end_step,
            )
        ),
        errors,
    )
    step_memory = _read_family(
        "step_memory",
        lambda: read_step_memory_measurements(
            connection,
            analysis_window=selection,
        ),
        errors,
    )
    return RecentMeasurements(
        system=system,
        process=process,
        step_time=step_time,
        step_memory=step_memory,
        errors=errors,
    )


__all__ = ["RecentMeasurements", "read_recent"]
