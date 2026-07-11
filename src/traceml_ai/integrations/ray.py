# Copyright 2026 OptAI UG (haftungsbeschraenkt)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Ray Train integration for TraceML.

Ray remains responsible for scheduling, worker startup, rank setup, and
distributed training communication. TraceML starts one lightweight aggregator
actor and one TraceML runtime inside each Ray Train worker. Worker runtimes send
telemetry to the aggregator over the same TCP path used by the CLI launcher.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional

from traceml_ai.reporting.config import DEFAULT_SUMMARY_WINDOW_ROWS
from traceml_ai.runtime.lifecycle import RuntimeHandle, start_aggregator
from traceml_ai.runtime.lifecycle import start_runtime as start_runtime_handle
from traceml_ai.runtime.settings import (
    AggregatorEndpoint,
    AggregatorTransportSettings,
    TraceMLSettings,
)

TrainLoop = Callable[[Dict[str, Any]], Any]


@dataclass(frozen=True)
class TraceMLRayConfig:
    """TraceML settings used by the Ray Train integration.

    Parameters
    ----------
    mode:
        TraceML display/reporting mode. ``"summary"`` is the default for Ray
        because distributed worker logs are often noisy.
    profile:
        TraceML sampler profile. Public Ray integration uses the normal
        ``"run"`` profile.
    init_mode:
        Instrumentation mode passed to ``traceml.init()`` inside each worker.
    patch_dataloader:
        Selective-mode-only override for DataLoader fetch patching.
    patch_forward:
        Selective-mode-only override for forward-pass patching.
    patch_backward:
        Selective-mode-only override for backward-pass patching.
    patch_h2d:
        Selective-mode-only override for host-to-device transfer patching.
    logs_dir:
        Directory where TraceML writes session logs and summary artifacts.
    session_id:
        Optional explicit TraceML session id. If omitted, a unique Ray session
        id is generated for each ``fit()`` call.
    sampler_interval_sec:
        Background sampler cadence in seconds.
    summary_window_rows:
        Number of recent history rows used by final summary generation.
    bind_host:
        Host interface used by the aggregator actor. Use ``"0.0.0.0"`` for
        multi-node Ray clusters so workers on other nodes can connect.
    port:
        Aggregator TCP port. ``0`` asks the OS to pick a free port.
    stop_timeout_sec:
        Best-effort timeout for aggregator shutdown.
    """

    mode: str = "summary"
    profile: str = "run"
    init_mode: str = "auto"
    patch_dataloader: Optional[bool] = None
    patch_forward: Optional[bool] = None
    patch_backward: Optional[bool] = None
    patch_h2d: Optional[bool] = None
    logs_dir: str = "./logs"
    session_id: str = ""
    sampler_interval_sec: float = 1.0
    summary_window_rows: int = DEFAULT_SUMMARY_WINDOW_ROWS
    bind_host: str = "0.0.0.0"
    port: int = 0
    stop_timeout_sec: float = 5.0


def _require_ray() -> tuple[Any, Any]:
    """Import Ray only when the Ray integration is actually used."""
    try:
        import ray
        from ray.train.torch import TorchTrainer
    except ImportError as exc:
        raise ImportError(
            "TraceML Ray integration requires Ray. Install it with "
            "`pip install 'traceml-ai[ray]'`."
        ) from exc
    return ray, TorchTrainer


def _new_session_id() -> str:
    """Return a readable unique TraceML session id for one Ray fit call."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"ray_{stamp}_{suffix}"


def _normalize_config(config: Optional[TraceMLRayConfig]) -> TraceMLRayConfig:
    """Return a Ray config with a concrete session id."""
    current = config or TraceMLRayConfig()
    session_id = str(current.session_id or "").strip()
    if session_id:
        return current
    return replace(current, session_id=_new_session_id())


def _endpoint_from_mapping(endpoint: Mapping[str, Any]) -> AggregatorEndpoint:
    """Validate and convert a Ray-serializable endpoint mapping."""
    return AggregatorEndpoint(
        host=str(endpoint["host"]),
        port=int(endpoint["port"]),
        session_id=str(endpoint["session_id"]),
    )


def _build_aggregator_settings(
    *,
    config: TraceMLRayConfig,
    connect_host: str,
) -> TraceMLSettings:
    """Build settings for the TraceML aggregator actor process."""
    return TraceMLSettings(
        mode=str(config.mode),
        profile=str(config.profile),
        sampler_interval_sec=float(config.sampler_interval_sec),
        logs_dir=str(config.logs_dir),
        session_id=str(config.session_id),
        summary_window_rows=int(config.summary_window_rows),
        aggregator=AggregatorTransportSettings(
            connect_host=str(connect_host),
            bind_host=str(config.bind_host),
            port=int(config.port),
        ),
    )


def _build_worker_settings(
    *,
    config: TraceMLRayConfig,
    endpoint: AggregatorEndpoint,
) -> TraceMLSettings:
    """Build settings for one TraceML runtime inside a Ray Train worker."""
    return TraceMLSettings(
        mode=str(config.mode),
        profile=str(config.profile),
        sampler_interval_sec=float(config.sampler_interval_sec),
        logs_dir=str(config.logs_dir),
        session_id=str(endpoint.session_id),
        summary_window_rows=int(config.summary_window_rows),
        aggregator=AggregatorTransportSettings(
            connect_host=str(endpoint.host),
            bind_host=str(config.bind_host),
            port=int(endpoint.port),
        ),
    )


def _apply_ray_train_identity_env() -> None:
    """Mirror Ray Train worker identity into torchrun-style environment vars.

    TraceML's runtime identity resolver already understands ``RANK``,
    ``WORLD_SIZE``, ``LOCAL_RANK``, ``LOCAL_WORLD_SIZE``, and node rank
    variables. Ray Train exposes the same concepts through its context object,
    so this bridge keeps the core runtime launcher-agnostic.
    """
    try:
        from ray import train

        context = train.get_context()
    except Exception:
        return

    methods = {
        "RANK": "get_world_rank",
        "WORLD_SIZE": "get_world_size",
        "LOCAL_RANK": "get_local_rank",
        "LOCAL_WORLD_SIZE": "get_local_world_size",
        "NODE_RANK": "get_node_rank",
    }
    for env_name, method_name in methods.items():
        method = getattr(context, method_name, None)
        if method is None:
            continue
        try:
            value = method()
        except Exception:
            continue
        if value is not None:
            os.environ[env_name] = str(value)


class _TraceMLAggregatorActor:
    """Ray actor implementation that owns one TraceML aggregator lifecycle."""

    def __init__(self, config: TraceMLRayConfig) -> None:
        import ray

        self._config = config
        self._closed = False
        self._node_ip = str(ray.util.get_node_ip_address())
        self._handle = start_aggregator(
            _build_aggregator_settings(
                config=config,
                connect_host=self._node_ip,
            )
        )

    def endpoint(self) -> dict[str, Any]:
        """Return the reachable aggregator endpoint for Ray Train workers."""
        endpoint = self._handle.endpoint
        return {
            "host": str(self._node_ip),
            "port": int(endpoint.port),
            "session_id": str(endpoint.session_id),
        }

    def stop(self) -> None:
        """Stop the TraceML aggregator once."""
        if self._closed:
            return
        self._closed = True
        self._handle.stop(timeout_sec=float(self._config.stop_timeout_sec))


@dataclass(frozen=True)
class _TraceMLWorkerLoop:
    """Callable Ray Train loop wrapper that owns worker runtime lifecycle."""

    train_loop_per_worker: TrainLoop
    endpoint: AggregatorEndpoint
    config: TraceMLRayConfig

    def __call__(self, train_loop_config: Dict[str, Any]) -> Any:
        import traceml_ai as traceml

        _apply_ray_train_identity_env()

        settings = _build_worker_settings(
            config=self.config,
            endpoint=self.endpoint,
        )
        handle: Optional[RuntimeHandle] = None
        try:
            handle = start_runtime_handle(settings, fail_open=True)
            traceml.init(
                mode=str(self.config.init_mode),
                patch_dataloader=self.config.patch_dataloader,
                patch_forward=self.config.patch_forward,
                patch_backward=self.config.patch_backward,
                patch_h2d=self.config.patch_h2d,
            )
            return self.train_loop_per_worker(train_loop_config)
        finally:
            if handle is not None:
                handle.stop()


def _stop_actor_best_effort(ray: Any, actor: Any) -> None:
    """Stop and kill a Ray actor without masking the caller's original error."""
    try:
        ray.get(actor.stop.remote())
    except Exception:
        pass

    try:
        ray.kill(actor, no_restart=True)
    except Exception:
        pass


class TraceMLTorchTrainer:
    """TraceML wrapper for ``ray.train.torch.TorchTrainer``.

    The wrapper deliberately uses composition instead of subclassing. Ray's
    ``TorchTrainer`` still owns training orchestration; TraceML only adds:

    - one aggregator actor before ``fit()``
    - one runtime wrapper inside each worker
    - best-effort aggregator shutdown after ``fit()`` completes or fails
    """

    def __init__(
        self,
        train_loop_per_worker: TrainLoop,
        *,
        train_loop_config: Optional[Dict[str, Any]] = None,
        traceml_config: Optional[TraceMLRayConfig] = None,
        **torch_trainer_kwargs: Any,
    ) -> None:
        self._train_loop_per_worker = train_loop_per_worker
        self._train_loop_config = dict(train_loop_config or {})
        self._traceml_config = traceml_config or TraceMLRayConfig()
        self._torch_trainer_kwargs = dict(torch_trainer_kwargs)
        self._last_endpoint: Optional[AggregatorEndpoint] = None

    @property
    def last_endpoint(self) -> Optional[AggregatorEndpoint]:
        """Aggregator endpoint used by the most recent ``fit()`` call."""
        return self._last_endpoint

    def fit(self) -> Any:
        """Run Ray Train with TraceML telemetry enabled."""
        ray, TorchTrainer = _require_ray()

        if os.environ.get("TRACEML_DISABLED") == "1":
            trainer = TorchTrainer(
                self._train_loop_per_worker,
                train_loop_config=dict(self._train_loop_config),
                **self._torch_trainer_kwargs,
            )
            return trainer.fit()

        config = _normalize_config(self._traceml_config)

        Actor = ray.remote(num_cpus=1)(_TraceMLAggregatorActor)
        actor = Actor.remote(config)

        try:
            endpoint = _endpoint_from_mapping(ray.get(actor.endpoint.remote()))
            self._last_endpoint = endpoint
            wrapped_loop = _TraceMLWorkerLoop(
                train_loop_per_worker=self._train_loop_per_worker,
                endpoint=endpoint,
                config=config,
            )
            trainer = TorchTrainer(
                wrapped_loop,
                train_loop_config=dict(self._train_loop_config),
                **self._torch_trainer_kwargs,
            )
            return trainer.fit()
        finally:
            _stop_actor_best_effort(ray, actor)


__all__ = [
    "TraceMLRayConfig",
    "TraceMLTorchTrainer",
]
