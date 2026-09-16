import importlib
import sys
from unittest.mock import Mock

import pytest

from traceml_ai.integrations.ray import (
    TraceMLRayConfig,
    _build_aggregator_settings,
    _build_worker_settings,
    _endpoint_from_mapping,
    _normalize_config,
)
from traceml_ai.runtime.settings import AggregatorEndpoint


def test_ray_integration_module_import_is_lazy():
    """
    Importing the TraceML Ray module should not import Ray itself.

    Ray is an optional dependency, so users without Ray installed must still be
    able to import the rest of TraceML and inspect integration modules.
    """
    sys.modules.pop("traceml_ai.integrations.ray", None)
    sys.modules.pop("ray", None)

    importlib.import_module("traceml_ai.integrations.ray")

    assert "ray" not in sys.modules


def test_normalize_config_generates_session_id():
    config = _normalize_config(TraceMLRayConfig())

    assert config.session_id.startswith("ray_")
    assert config.mode == "summary"
    assert config.port == 0
    assert config.sampler_interval_sec == 2.0


def test_normalize_config_preserves_explicit_session_id():
    config = _normalize_config(TraceMLRayConfig(session_id="run-123"))

    assert config.session_id == "run-123"


def test_endpoint_from_mapping_validates_types():
    endpoint = _endpoint_from_mapping(
        {
            "host": "10.0.0.4",
            "port": "12345",
            "session_id": "ray-run",
        }
    )

    assert endpoint == AggregatorEndpoint(
        host="10.0.0.4",
        port=12345,
        session_id="ray-run",
    )


def test_aggregator_settings_use_actor_node_as_connect_host():
    settings = _build_aggregator_settings(
        config=TraceMLRayConfig(session_id="ray-run", port=0),
        connect_host="10.0.0.9",
    )

    assert settings.session_id == "ray-run"
    assert settings.mode == "summary"
    assert settings.aggregator.connect_host == "10.0.0.9"
    assert settings.aggregator.bind_host == "0.0.0.0"
    assert settings.aggregator.port == 0
    assert settings.render_interval_sec == 2.0


def test_worker_settings_connect_to_actor_endpoint():
    settings = _build_worker_settings(
        config=TraceMLRayConfig(
            session_id="ignored-on-worker",
            sampler_interval_sec=2.0,
        ),
        endpoint=AggregatorEndpoint(
            host="10.0.0.9",
            port=34567,
            session_id="actor-session",
        ),
    )

    assert settings.session_id == "actor-session"
    assert settings.sampler_interval_sec == 2.0
    assert settings.aggregator.connect_host == "10.0.0.9"
    assert settings.aggregator.port == 34567


def test_ray_actor_logs_finalization_failure_once(monkeypatch):
    import traceml_ai.integrations.ray as ray_integration

    failure = RuntimeError("summary generation failed")
    handle = Mock()
    handle.stop.side_effect = failure
    logger = Mock()
    fake_ray = Mock()
    fake_ray.util.get_node_ip_address.return_value = "10.0.0.9"
    start_aggregator = Mock(return_value=handle)
    get_error_logger = Mock(return_value=logger)
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(ray_integration, "start_aggregator", start_aggregator)
    monkeypatch.setattr(ray_integration, "get_error_logger", get_error_logger)

    actor = ray_integration._TraceMLAggregatorActor(
        TraceMLRayConfig(session_id="ray-run", stop_timeout_sec=7.5)
    )

    with pytest.raises(RuntimeError) as raised:
        actor.stop()

    assert raised.value is failure
    start_aggregator.assert_called_once()
    get_error_logger.assert_called_once_with("TraceMLRayAggregator")
    handle.stop.assert_called_once_with(timeout_sec=7.5)
    logger.error.assert_called_once()
    log_message = logger.error.call_args.args[0]
    assert "[TraceML] Ray aggregator finalization failed" in log_message
    assert "RuntimeError: summary generation failed" in log_message

    # A failed stop is still terminal; cleanup must not log or stop twice.
    actor.stop()
    handle.stop.assert_called_once()
    logger.error.assert_called_once()


def test_ray_actor_stop_remains_best_effort_at_driver_boundary():
    import traceml_ai.integrations.ray as ray_integration

    actor = Mock()
    actor.stop.remote.return_value = "stop-ref"
    ray = Mock()
    ray.get.side_effect = RuntimeError("remote finalization failed")

    ray_integration._stop_actor_best_effort(ray, actor)

    ray.get.assert_called_once_with("stop-ref")
    ray.kill.assert_called_once_with(actor, no_restart=True)


def test_ray_disabled_delegates_to_native_torch_trainer(monkeypatch):
    import traceml_ai.integrations.ray as ray_integration

    observed = {}

    class _FakeRay:
        def remote(self, *args, **kwargs):
            raise AssertionError("disabled Ray path must not create actors")

    class _FakeTorchTrainer:
        def __init__(
            self,
            train_loop_per_worker,
            *,
            train_loop_config=None,
            **kwargs,
        ):
            observed["train_loop_per_worker"] = train_loop_per_worker
            observed["train_loop_config"] = dict(train_loop_config or {})
            observed["kwargs"] = dict(kwargs)

        def fit(self):
            return self._run_native()

        def _run_native(self):
            return observed["train_loop_per_worker"](
                observed["train_loop_config"]
            )

    def _train_loop(config):
        return {"native": True, "config": config}

    monkeypatch.setenv("TRACEML_DISABLED", "1")
    monkeypatch.setattr(
        ray_integration,
        "_require_ray",
        lambda: (_FakeRay(), _FakeTorchTrainer),
    )

    trainer = ray_integration.TraceMLTorchTrainer(
        _train_loop,
        train_loop_config={"batch_size": 4},
        scaling_config="native-scaling",
    )

    assert trainer.fit() == {
        "native": True,
        "config": {"batch_size": 4},
    }
    assert not isinstance(
        observed["train_loop_per_worker"],
        ray_integration._TraceMLWorkerLoop,
    )
    assert observed["kwargs"] == {"scaling_config": "native-scaling"}
    assert trainer.last_endpoint is None
