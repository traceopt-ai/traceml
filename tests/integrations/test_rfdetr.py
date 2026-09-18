"""RF-DETR factory integration tests without an RF-DETR installation."""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import warnings
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from traceml_ai.integrations import lightning
from traceml_ai.integrations import rfdetr


@pytest.fixture
def rf_factory(monkeypatch):
    """Provide the documented RF-DETR factory boundary and its defaults."""
    training = ModuleType("rfdetr.training")
    calls = []
    default_callbacks = [object(), object(), object()]
    trainer = SimpleNamespace(
        callbacks=list(default_callbacks),
        strategy=SimpleNamespace(root_device="cpu"),
    )

    def original(
        train_config,
        model_config,
        *,
        accelerator=None,
        include_training_callbacks=True,
        **trainer_kwargs,
    ):
        calls.append(
            (
                train_config,
                model_config,
                accelerator,
                include_training_callbacks,
                trainer_kwargs,
            )
        )
        return trainer

    training.build_trainer = original
    monkeypatch.setitem(sys.modules, "rfdetr.training", training)
    monkeypatch.setattr(lightning, "IS_LIGHTNING_AVAILABLE", True)
    effective = SimpleNamespace(disabled=False)
    monkeypatch.setattr(lightning, "init", lambda: effective)
    monkeypatch.setattr(rfdetr, "version", lambda package: "1.10.1")
    return SimpleNamespace(
        training=training,
        original=original,
        trainer=trainer,
        defaults=default_callbacks,
        calls=calls,
        effective=effective,
        train_config=SimpleNamespace(accelerator="cpu", strategy="auto"),
        model_config=SimpleNamespace(
            compile=False,
            segmentation_head=False,
            use_grouppose_keypoints=False,
        ),
    )


def test_rfdetr_module_import_does_not_load_training_dependencies():
    source = Path(__file__).resolve().parents[2] / "src"
    env = {**os.environ, "PYTHONPATH": str(source)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "class BlockTrainingImports:\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname.split('.')[0] in "
            "{'rfdetr', 'torch', 'lightning', 'pytorch_lightning'}:\n"
            "            raise AssertionError(fullname)\n"
            "sys.meta_path.insert(0, BlockTrainingImports())\n"
            "from traceml_ai.integrations import rfdetr\n"
            "assert callable(rfdetr.init)\n",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_init_preserves_defaults_factory_signature_and_arguments(rf_factory):
    assert rfdetr.init() is rf_factory.effective
    assert inspect.signature(rf_factory.training.build_trainer) == (
        inspect.signature(rf_factory.original)
    )
    returned = rf_factory.training.build_trainer(
        train_config=rf_factory.train_config,
        model_config=rf_factory.model_config,
        accelerator="cpu",
        devices=2,
        num_nodes=2,
        strategy="ddp",
        limit_train_batches=3,
    )
    assert returned is rf_factory.trainer
    assert returned.callbacks[:-1] == rf_factory.defaults
    assert isinstance(returned.callbacks[-1], rfdetr._callback_class())
    assert rf_factory.calls == [
        (
            rf_factory.train_config,
            rf_factory.model_config,
            "cpu",
            True,
            {
                "devices": 2,
                "num_nodes": 2,
                "strategy": "ddp",
                "limit_train_batches": 3,
            },
        )
    ]


def test_init_is_idempotent_and_existing_rf_callback_is_not_duplicated(
    rf_factory,
):
    rfdetr.init()
    wrapped = rf_factory.training.build_trainer
    rfdetr.init()
    assert rf_factory.training.build_trainer is wrapped
    wrapped(rf_factory.train_config, rf_factory.model_config)
    wrapped(rf_factory.train_config, rf_factory.model_config)
    assert len(rf_factory.calls) == 2
    assert len(rf_factory.trainer.callbacks) == len(rf_factory.defaults) + 1


def test_factory_leaves_evaluation_unchanged_even_for_unsupported_model(
    rf_factory,
):
    rfdetr.init()
    rf_factory.model_config.segmentation_head = True
    returned = rf_factory.training.build_trainer(
        rf_factory.train_config,
        rf_factory.model_config,
        include_training_callbacks=False,
    )
    assert returned is rf_factory.trainer
    assert returned.callbacks == rf_factory.defaults


def test_disabled_init_does_not_import_rf_or_install_hook(
    rf_factory, monkeypatch
):
    monkeypatch.setenv("TRACEML_DISABLED", "1")

    def unexpected_import(name):
        raise AssertionError(name)

    monkeypatch.setattr(rfdetr.importlib, "import_module", unexpected_import)
    assert rfdetr.init() is rf_factory.effective
    assert rf_factory.training.build_trainer is rf_factory.original


def test_fail_open_init_does_not_install_factory(rf_factory):
    rf_factory.effective.disabled = True
    assert rfdetr.init() is rf_factory.effective
    assert rf_factory.training.build_trainer is rf_factory.original


def test_factory_honors_disabled_after_init(rf_factory, monkeypatch):
    rfdetr.init()
    monkeypatch.setenv("TRACEML_DISABLED", "1")
    rf_factory.model_config.compile = True
    returned = rf_factory.training.build_trainer(
        rf_factory.train_config, rf_factory.model_config
    )
    assert returned.callbacks == rf_factory.defaults


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("model_config", "compile", True, "compiled training"),
        ("model_config", "segmentation_head", True, "segmentation"),
        ("model_config", "use_grouppose_keypoints", True, "keypoint"),
        ("model_config", "cuda_graphs", True, "CUDA graphs"),
        ("train_config", "strategy", "ddp_spawn", "ddp_spawn"),
        ("train_config", "strategy", "ddp_notebook", "ddp_notebook"),
        ("train_config", "strategy", "ddp_fork", "ddp_fork"),
        ("train_config", "strategy", "fsdp", "fsdp"),
        ("train_config", "accelerator", "mps", "mps"),
        ("train_config", "accelerator", "tpu", "tpu"),
    ],
)
def test_unsupported_modes_warn_and_preserve_native_trainer(
    rf_factory, capsys, target, field, value, message
):
    rfdetr.init()
    setattr(getattr(rf_factory, target), field, value)
    returned = rf_factory.training.build_trainer(
        rf_factory.train_config, rf_factory.model_config
    )
    assert returned is rf_factory.trainer
    assert len(rf_factory.calls) == 1
    assert rf_factory.trainer.callbacks == rf_factory.defaults
    error = capsys.readouterr().err
    assert "[TraceML] RF-DETR: skipping instrumentation" in error
    assert message in error


def test_trainer_keyword_overrides_are_checked(rf_factory, capsys):
    rfdetr.init()
    returned = rf_factory.training.build_trainer(
        rf_factory.train_config,
        rf_factory.model_config,
        strategy="deepspeed",
    )
    assert returned.callbacks == rf_factory.defaults
    assert rf_factory.calls[0][-1] == {"strategy": "deepspeed"}
    assert "deepspeed" in capsys.readouterr().err


@pytest.mark.parametrize(
    "field", ["compile", "segmentation_head", "use_grouppose_keypoints"]
)
def test_missing_mode_field_skips_instrumentation(rf_factory, capsys, field):
    delattr(rf_factory.model_config, field)
    rfdetr.init()
    returned = rf_factory.training.build_trainer(
        rf_factory.train_config, rf_factory.model_config
    )
    assert returned.callbacks == rf_factory.defaults
    assert len(rf_factory.calls) == 1
    assert field in capsys.readouterr().err


@pytest.mark.parametrize("start_method", ["popen", "spawn", "fork"])
def test_explicit_ddp_strategy_requires_ordinary_launcher(
    rf_factory, capsys, start_method
):
    strategies = pytest.importorskip("pytorch_lightning.strategies")
    strategy = strategies.DDPStrategy(start_method=start_method)
    rfdetr.init()
    returned = rf_factory.training.build_trainer(
        rf_factory.train_config,
        rf_factory.model_config,
        strategy=strategy,
    )
    assert len(rf_factory.calls) == 1
    if start_method == "popen":
        assert isinstance(returned.callbacks[-1], rfdetr._callback_class())
    else:
        assert returned.callbacks == rf_factory.defaults
        assert "ordinary DDP" in capsys.readouterr().err


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_supported_resolved_device(rf_factory, device):
    rf_factory.trainer.strategy.root_device = device
    rfdetr.init()
    returned = rf_factory.training.build_trainer(
        rf_factory.train_config, rf_factory.model_config
    )
    assert isinstance(returned.callbacks[-1], rfdetr._callback_class())


def test_auto_accelerator_cannot_silently_trace_unsupported_device(
    rf_factory, capsys
):
    rf_factory.train_config.accelerator = "auto"
    rf_factory.trainer.strategy.root_device = "mps:0"
    rfdetr.init()
    returned = rf_factory.training.build_trainer(
        rf_factory.train_config, rf_factory.model_config
    )
    assert returned is rf_factory.trainer
    assert len(rf_factory.calls) == 1
    assert rf_factory.trainer.callbacks == rf_factory.defaults
    assert "unsupported resolved device 'mps:0'" in capsys.readouterr().err


def test_generic_callback_conflict_is_actionable_and_does_not_mutate(
    rf_factory, capsys
):
    generic = lightning.TraceMLCallback()
    rf_factory.trainer.callbacks.append(generic)
    before = list(rf_factory.trainer.callbacks)
    rfdetr.init()
    returned = rf_factory.training.build_trainer(
        rf_factory.train_config, rf_factory.model_config
    )
    assert returned is rf_factory.trainer
    assert rf_factory.trainer.callbacks == before
    assert "existing generic or duplicate" in capsys.readouterr().err


def test_duplicate_rf_callbacks_warn_without_adding_another(
    rf_factory, capsys
):
    rf_factory.trainer.callbacks.extend(
        [rfdetr._callback_class()(), rfdetr._callback_class()()]
    )
    before = list(rf_factory.trainer.callbacks)
    rfdetr.init()
    returned = rf_factory.training.build_trainer(
        rf_factory.train_config, rf_factory.model_config
    )
    assert returned.callbacks == before
    assert "duplicate TraceML callbacks" in capsys.readouterr().err


def test_callback_selects_inner_model(rf_factory):
    inner = object()
    assert (
        rfdetr._callback_class()()._forward_target(
            SimpleNamespace(model=inner)
        )
        is inner
    )


def test_missing_rf_dependency_has_install_guidance(rf_factory, monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("No module named 'rfdetr'", name="rfdetr")

    monkeypatch.setattr(rfdetr.importlib, "import_module", missing)
    with pytest.raises(ImportError, match=r"rfdetr\[train\]==1.10.1"):
        rfdetr.init()
    assert rf_factory.training.build_trainer is rf_factory.original


def test_internal_rf_import_error_is_not_misreported(rf_factory, monkeypatch):
    error = ModuleNotFoundError("internal failure", name="rfdetr.internal")

    def missing(name):
        raise error

    monkeypatch.setattr(rfdetr.importlib, "import_module", missing)
    with pytest.raises(ModuleNotFoundError) as captured:
        rfdetr.init()
    assert captured.value is error


def test_incompatible_factory_warns_without_patching(rf_factory, capsys):
    def unsupported():
        return None

    rf_factory.training.build_trainer = unsupported
    assert rfdetr.init() is rf_factory.effective
    assert rf_factory.training.build_trainer is unsupported
    assert "unsupported build_trainer interface" in capsys.readouterr().err


@pytest.mark.parametrize("unsupported", [False, True])
def test_factory_error_propagates_unchanged(rf_factory, unsupported):
    error = RuntimeError("training configuration failed")
    calls = []

    def failing(
        train_config,
        model_config,
        *,
        accelerator=None,
        include_training_callbacks=True,
        **trainer_kwargs,
    ):
        calls.append(True)
        raise error

    rf_factory.training.build_trainer = failing
    rf_factory.model_config.segmentation_head = unsupported
    rfdetr.init()
    with pytest.raises(RuntimeError) as captured:
        rf_factory.training.build_trainer(
            rf_factory.train_config, rf_factory.model_config
        )
    assert captured.value is error
    assert calls == [True]


def test_factory_setup_failure_warns_without_patching(
    rf_factory, monkeypatch, capsys
):
    def unavailable():
        raise RuntimeError("callback unavailable")

    monkeypatch.setattr(rfdetr, "_callback_class", unavailable)
    assert rfdetr.init() is rf_factory.effective
    assert rf_factory.training.build_trainer is rf_factory.original
    assert "callback unavailable" in capsys.readouterr().err


def test_qualified_release_has_no_version_warning(rf_factory, capsys):
    rfdetr.init()
    assert not capsys.readouterr().err


@pytest.mark.parametrize("installed_version", ["1.11.0.dev0", "1.12.0"])
def test_unqualified_version_warns_once_without_blocking(
    rf_factory, monkeypatch, capsys, installed_version
):
    monkeypatch.setattr(rfdetr, "version", lambda package: installed_version)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rfdetr.init()
        assert (
            "[TraceML] RF-DETR: version " + installed_version
            in capsys.readouterr().err
        )
        rfdetr.init()
    assert not capsys.readouterr().err


def test_source_checkout_without_package_metadata_warns(
    rf_factory, monkeypatch, capsys
):
    def missing_version(package):
        raise rfdetr.PackageNotFoundError(package)

    monkeypatch.setattr(rfdetr, "version", missing_version)
    rfdetr.init()
    assert "[TraceML] RF-DETR: version unknown" in capsys.readouterr().err
