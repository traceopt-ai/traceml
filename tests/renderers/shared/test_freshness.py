# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Telemetry freshness, and why it is not the cached-payload TTL."""

from __future__ import annotations

import pytest

from traceml_ai.renderers.shared.freshness import (
    DEFAULT_STALE_TICKS,
    MIN_STALE_AFTER_S,
    CachedPayloadTTL,
    FreshnessPolicy,
)


# --- the threshold follows the cadence -----------------------------------
def test_a_slower_sampler_gets_a_longer_patience():
    fast = FreshnessPolicy(interval_s=2.0)
    slow = FreshnessPolicy(interval_s=30.0)
    assert fast.stale_after_s == pytest.approx(6.0)
    assert slow.stale_after_s == pytest.approx(90.0)


def test_a_very_fast_sampler_does_not_flag_on_scheduler_noise():
    """Three ticks of a 100 ms sampler is 0.3 s, which any hiccup exceeds."""
    policy = FreshnessPolicy(interval_s=0.1)
    assert policy.stale_after_s == pytest.approx(MIN_STALE_AFTER_S)


def test_the_default_patience_is_three_ticks():
    assert DEFAULT_STALE_TICKS == 3.0
    assert FreshnessPolicy(interval_s=10.0).stale_after_s == pytest.approx(
        30.0
    )


# --- building the policy -------------------------------------------------
def test_the_configured_sampler_interval_is_the_input():
    assert FreshnessPolicy.from_interval(5.0).interval_s == pytest.approx(5.0)


@pytest.mark.parametrize("bad", [None, 0.0])
def test_a_missing_interval_falls_back_once_in_one_place(bad):
    assert FreshnessPolicy.from_interval(bad).interval_s == pytest.approx(2.0)


def test_the_real_default_sampler_interval_is_usable_here():
    """Wired against the setting rather than a number retyped locally."""
    from traceml_ai.runtime.settings import TraceMLSettings

    configured = TraceMLSettings().sampler_interval_sec
    policy = FreshnessPolicy.from_interval(configured)
    assert policy.interval_s == pytest.approx(float(configured))
    assert policy.stale_after_s >= MIN_STALE_AFTER_S


def test_an_observed_cadence_beats_what_was_configured():
    """A rank sampling slower than requested is judged on its real rhythm."""
    policy = FreshnessPolicy.from_observed_cadence(9.0, configured_s=2.0)
    assert policy.interval_s == pytest.approx(9.0)


def test_the_configured_interval_is_the_fallback_when_nothing_observed():
    policy = FreshnessPolicy.from_observed_cadence(None, configured_s=7.0)
    assert policy.interval_s == pytest.approx(7.0)


# --- the judgement -------------------------------------------------------
def test_a_rank_inside_its_patience_is_live():
    policy = FreshnessPolicy(interval_s=10.0)
    assert policy.is_stale(29.0) is False


def test_a_rank_past_its_patience_is_stale():
    policy = FreshnessPolicy(interval_s=10.0)
    assert policy.is_stale(31.0) is True


def test_an_unknown_age_is_not_evidence_of_death():
    """Missing information is not a dead rank; saying so would invent one."""
    assert FreshnessPolicy(interval_s=2.0).is_stale(None) is False


def test_age_is_clamped_so_clock_skew_does_not_read_as_fresher_than_fresh():
    policy = FreshnessPolicy(interval_s=2.0)
    assert policy.age_of(120.0, now_s=100.0) == pytest.approx(0.0)
    assert policy.age_of(90.0, now_s=100.0) == pytest.approx(10.0)
    assert policy.age_of(None, now_s=100.0) is None


# --- the separation ------------------------------------------------------
def test_a_cached_payload_ttl_is_a_different_type_entirely():
    """The two questions were one ambiguous word before this module.

    Reusing a cached payload says the DATABASE could not be read. It says
    nothing about whether the ranks are alive, and the types are separate
    so a caller cannot pass one where it meant the other.
    """
    assert not isinstance(CachedPayloadTTL(), FreshnessPolicy)
    assert not hasattr(CachedPayloadTTL(), "is_stale")
    assert not hasattr(FreshnessPolicy(interval_s=2.0), "may_reuse")


def test_a_cached_payload_may_be_reused_inside_its_ttl():
    ttl = CachedPayloadTTL(ttl_s=30.0)
    assert ttl.may_reuse(10.0) is True
    assert ttl.may_reuse(30.0) is True
    assert ttl.may_reuse(31.0) is False


def test_a_ttl_of_none_never_expires():
    assert CachedPayloadTTL(ttl_s=None).may_reuse(1e9) is True


def test_a_long_cache_ttl_does_not_make_a_dead_rank_look_live():
    """The regression the split exists to prevent.

    A generous cached-payload TTL must not extend a rank's freshness: the
    rank is 10 minutes silent and stays stale no matter how long a stale
    payload may be served.
    """
    ttl = CachedPayloadTTL(ttl_s=3600.0)
    policy = FreshnessPolicy(interval_s=2.0)
    assert ttl.may_reuse(600.0) is True
    assert policy.is_stale(600.0) is True
