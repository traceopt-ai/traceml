"""Exercise the public RF-DETR API with a real, offline Nano detector."""

import json
import socket

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rfdetr")

from rfdetr.training.callbacks.ema import RFDETREMACallback  # noqa: E402

from traceml_ai.instrumentation.hooks.optimizer_hooks import (  # noqa: E402
    reset_optimizer_timing,
)
from traceml_ai.instrumentation.step_events import (  # noqa: E402
    abort_step_capture,
    begin_step_capture,
    drain_step_memory_events,
    drain_step_time_batches,
)
from traceml_ai.integrations import rfdetr as tracing  # noqa: E402
from traceml_ai.runtime.state import (  # noqa: E402
    configure_trace_recording,
    reset_trace_session_state,
)


@pytest.fixture(autouse=True)
def _isolation(monkeypatch):
    import rfdetr.training

    monkeypatch.setattr(
        rfdetr.training, "build_trainer", rfdetr.training.build_trainer
    )
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    reset_optimizer_timing()
    reset_trace_session_state()
    configure_trace_recording(max_steps=None)
    drain_step_time_batches()
    drain_step_memory_events()
    abort_step_capture(begin_step_capture())
    yield
    abort_step_capture(begin_step_capture())
    drain_step_time_batches()
    drain_step_memory_events()
    reset_optimizer_timing()
    torch.set_num_threads(threads)


def counts(batches, phase):
    return [
        sum(
            event.name == "_traceml_internal:" + phase
            for event in batch.events
        )
        for batch in batches
    ]


def test_rfdetr_nano_public_train_offline(tmp_path, monkeypatch):
    """Real detector/data/EMA/checkpoints; no downloaded images or weights."""
    import requests
    import rfdetr.training
    from PIL import Image
    from rfdetr import RFDETRNano

    network_attempts = []

    def forbidden_network(*args, **kwargs):
        network_attempts.append(args)
        raise AssertionError("The RF-DETR smoke test must run offline")

    monkeypatch.setattr(
        requests.sessions.Session, "request", forbidden_network
    )
    monkeypatch.setattr(socket.socket, "connect", forbidden_network)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    for split in ("train", "valid", "test"):
        folder = tmp_path / "dataset" / split
        folder.mkdir(parents=True)
        Image.new("RGB", (64, 64), color=(80, 120, 160)).save(
            folder / "sample.jpg"
        )
        (folder / "_annotations.coco.json").write_text(
            json.dumps(
                {
                    "images": [
                        {
                            "id": 1,
                            "file_name": "sample.jpg",
                            "height": 64,
                            "width": 64,
                        }
                    ],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "category_id": 1,
                            "bbox": [8, 8, 24, 24],
                            "area": 576,
                            "iscrowd": 0,
                        }
                    ],
                    "categories": [{"id": 1, "name": "object"}],
                }
            )
        )
    tracing.init()
    factory = rfdetr.training.build_trainer
    trainers = []

    def bounded_factory(*args, **kwargs):
        fit = factory(
            *args,
            **kwargs,
            limit_train_batches=2,
            limit_val_batches=1,
            limit_test_batches=1,
            num_sanity_val_steps=1,
            enable_model_summary=False,
        )
        trainers.append(fit)
        return fit

    monkeypatch.setattr(rfdetr.training, "build_trainer", bounded_factory)
    # Lower resolution/query count only bounds CPU cost; all kernels are real.
    model = RFDETRNano(
        pretrain_weights=None,
        device="cpu",
        resolution=64,
        num_queries=4,
        num_select=4,
        group_detr=1,
        compile=False,
    )
    model.train(
        dataset_dir=str(tmp_path / "dataset"),
        dataset_file="roboflow",
        output_dir=str(tmp_path / "output"),
        epochs=1,
        batch_size=1,
        grad_accum_steps=1,
        num_workers=0,
        accelerator="cpu",
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        tensorboard=False,
        progress_bar=None,
        use_ema=True,
        run_test=True,
        save_dataset_grids=True,
        seed=13,
    )
    fit = trainers[0]
    assert (
        sum(
            isinstance(callback, tracing._callback_class())
            for callback in fit.callbacks
        )
        == 1
    )
    assert any(isinstance(cb, RFDETREMACallback) for cb in fit.callbacks)
    assert fit.global_step == 2
    batches = drain_step_time_batches()
    assert counts(batches, "forward_time") == [1, 1]
    assert counts(batches, "dataloader_next") == [1, 1]
    assert "forward" not in fit.lightning_module.model.__dict__
    assert (tmp_path / "output/checkpoint_best_total.pth").is_file()
    assert (tmp_path / "output/training_config.json").is_file()
    assert "test/mAP_50_95" in fit.callback_metrics
    model.predict(Image.new("RGB", (64, 64)), threshold=0.0)
    assert drain_step_time_batches() == []
    assert not network_attempts
