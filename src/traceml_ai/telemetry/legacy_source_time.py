# Copyright 2026 OptAI UG (haftungsbeschraenkt)
# SPDX-License-Identifier: Apache-2.0

"""Temporary source-time storage bridge; remove in RFC-0011 PR 5."""

LEGACY_MIGRATED_NS_TABLES = frozenset({"step_time_samples"})


def legacy_source_time_storage(table: str) -> tuple[str, int]:
    """Return the column and units per second for an internal table name.

    LEGACY SOURCE TIME: temporary backward compatibility for tables migrated
    one at a time; remove this module and its callers in RFC-0011 PR 5.
    """
    if table in LEGACY_MIGRATED_NS_TABLES:
        return "timestamp_unix_ns", 1_000_000_000
    return "sample_ts_s", 1
