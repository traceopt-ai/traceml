import importlib
import sys

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
