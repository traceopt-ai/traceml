# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Two different questions that were both being called "stale".

**Telemetry freshness** asks whether a rank or a node is still reporting.
It is answered from the sampling cadence: a rank that has missed several
ticks has stopped, and the card must say so rather than keep drawing its
last value as current.

**Cached-payload TTL** asks how long a previously good payload may stand in
for a read that just failed. It is answered from how long a reader will
tolerate an unchanged view, and it says nothing whatsoever about the health
of the run.

They were one ambiguous notion of "stale" spread across several modules,
each with its own hardcoded two-second assumption. They are separate types
here so a caller has to say which one it means, and so the sampler interval
enters as a value rather than a constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .run_series import finite

# Ticks a rank may miss before the block stops calling it live. Three is
# what the Process and System blocks converged on: one missed tick is
# ordinary jitter, three is a rank that stopped.
DEFAULT_STALE_TICKS = 3.0

# However fast the sampler is configured, nothing is called stale before
# this. Protects a sub-second cadence from flagging on scheduler noise.
MIN_STALE_AFTER_S = 5.0

FreshnessState = Literal["fresh", "stale", "unknown"]


@dataclass(frozen=True)
class FreshnessPolicy:
    """When telemetry from one entity stops counting as current.

    Built from the cadence the samples actually arrive at, so a run
    configured with a slow sampler is not permanently "stale" and a fast
    one is not judged against a stale-by-default constant.
    """

    interval_s: float
    stale_ticks: float = DEFAULT_STALE_TICKS
    min_stale_after_s: float = MIN_STALE_AFTER_S

    @classmethod
    def from_interval(
        cls,
        interval_s: Optional[float],
        *,
        fallback_s: float = 2.0,
    ) -> "FreshnessPolicy":
        """Build from a configured sampler interval.

        ``fallback_s`` is used only when the caller genuinely has no
        interval to offer, and is the one place a default cadence is
        written down.
        """
        usable = finite(interval_s)
        if usable is None or usable <= 0:
            usable = finite(fallback_s) or 2.0
        return cls(interval_s=max(usable, 1e-6))

    @classmethod
    def from_observed_cadence(
        cls,
        cadence_s: Optional[float],
        *,
        configured_s: Optional[float] = None,
    ) -> "FreshnessPolicy":
        """Prefer the cadence samples actually arrived at.

        What was configured is an intention; what arrived is a measurement,
        and a rank that samples slower than requested should be judged
        against its real rhythm.
        """
        observed = finite(cadence_s)
        if observed is not None and observed > 0:
            return cls(interval_s=observed)
        return cls.from_interval(configured_s)

    @property
    def stale_after_s(self) -> float:
        """Age at which telemetry stops being called current."""
        return max(
            self.min_stale_after_s,
            self.interval_s * self.stale_ticks,
        )

    def state_of(self, age_s: Optional[float]) -> FreshnessState:
        """Whether an entity is fresh, stale, or has an unknown age.

        Absence of a usable timestamp is missing information. Returning an
        explicit state prevents callers from treating it as either live or
        dead by accident.
        """
        usable = finite(age_s)
        if usable is None:
            return "unknown"
        return "stale" if usable > self.stale_after_s else "fresh"

    def age_of(
        self,
        sample_ts_s: Optional[float],
        *,
        now_s: float,
    ) -> Optional[float]:
        """Seconds since a sample, clamped at zero, or ``None`` if unknown.

        Clamped because clock skew between a rank and the aggregator can
        put a sample slightly in the future, and a negative age would read
        as fresher than fresh rather than as noise.
        """
        stamp = finite(sample_ts_s)
        now = finite(now_s)
        if stamp is None or now is None:
            return None
        return max(0.0, now - stamp)


@dataclass(frozen=True)
class CachedPayloadTTL:
    """How long a last-good payload may answer for a failed read.

    Deliberately a different type from :class:`FreshnessPolicy`. Returning
    a cached payload says the DATABASE could not be read just now; it never
    says the ranks are healthy. Conflating the two is how a dead rank keeps
    being drawn as live.
    """

    ttl_s: Optional[float] = 30.0

    def may_reuse(self, age_s: float) -> bool:
        """Whether a cached payload of this age may still be served."""
        if self.ttl_s is None:
            return True
        usable = finite(age_s)
        limit = finite(self.ttl_s)
        if usable is None or limit is None:
            return False
        return usable <= limit


__all__ = [
    "CachedPayloadTTL",
    "DEFAULT_STALE_TICKS",
    "FreshnessPolicy",
    "FreshnessState",
    "MIN_STALE_AFTER_S",
]
