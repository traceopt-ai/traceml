import tempfile
from pathlib import Path

import pytest

from traceml_ai.integrations.huggingface import (
    TraceMLTrainer,
    TraceMLTrainerCallback,
    init,
)

try:
    import torch
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
        Trainer,
        TrainingArguments,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class _TinyTokenizedDataset(
    torch.utils.data.Dataset if HAS_TRANSFORMERS else object
):
    """
    Small synthetic dataset for Trainer integration tests.

    This keeps the test deterministic and self-contained by avoiding external
    model or dataset downloads while still exercising the Hugging Face trainer
    stack with realistic tensor-shaped inputs.
    """

    def __init__(
        self,
        *,
        num_rows: int = 20,
        seq_len: int = 16,
        vocab_size: int = 128,
        num_labels: int = 4,
    ) -> None:
        self._rows = []
        for idx in range(int(num_rows)):
            token_ids = torch.arange(seq_len, dtype=torch.long) % vocab_size
            token_ids = token_ids + (idx % 7)
            self._rows.append(
                {
                    "input_ids": token_ids.clone(),
                    "attention_mask": torch.ones(seq_len, dtype=torch.long),
                    "labels": torch.tensor(idx % num_labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int):
        return self._rows[index]


def _build_tiny_model():
    return BertForSequenceClassification(
        BertConfig(
            vocab_size=128,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
            num_labels=4,
        )
    )


def _build_training_args(
    output_dir: str,
    *,
    max_steps: int,
    gradient_accumulation_steps: int = 1,
    batch_size: int = 4,
) -> "TrainingArguments":
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        logging_steps=1,
        use_cpu=not torch.cuda.is_available(),
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )


def _drain_step_memory_queue(model_id: int) -> list:
    """Drain only this test's StepMemoryEvents from the shared queue."""
    from traceml_ai.utils.step_memory import step_memory_queue

    drained = []
    leftover = []
    while not step_memory_queue.empty():
        evt = step_memory_queue.get_nowait()
        if getattr(evt, "model_id", None) == model_id:
            drained.append(evt)
        else:
            leftover.append(evt)

    # Put back unrelated events so we do not interfere with other tests.
    for evt in leftover:
        try:
            step_memory_queue.put_nowait(evt)
        except Exception:
            pass
    return drained


def _drain_step_time_queue() -> list:
    """Drain all StepTimeBatch entries from the shared queue."""
    from traceml_ai.utils.timing import get_step_time_queue

    queue = get_step_time_queue()
    batches = []
    while not queue.empty():
        batches.append(queue.get_nowait())
    return batches


def _reset_traceml_state() -> None:
    """Reset TraceML's process-local step counter and drain shared queues."""
    from traceml_ai.runtime.state import reset_trace_session_state

    reset_trace_session_state()
    _drain_step_time_queue()
    # Drain any leftover step-memory events. We don't filter by model_id here
    # because we want a clean slate; older tests' events would otherwise leak
    # into this test's drained count.
    from traceml_ai.utils.step_memory import step_memory_queue

    while not step_memory_queue.empty():
        try:
            step_memory_queue.get_nowait()
        except Exception:
            break


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_hf_trainer_integration():
    """
    Test that TraceMLTrainer (legacy thin-wrapper path) runs a few steps with
    a real model and TraceML instrumentation enabled.
    """
    _reset_traceml_state()
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "results"
        model = _build_tiny_model()
        train_dataset = _TinyTokenizedDataset()
        training_args = _build_training_args(str(output_dir), max_steps=5)

        trainer = TraceMLTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            traceml_enabled=True,
        )
        trainer.train()

        from traceml_ai.sdk.instrumentation import TraceState

        assert TraceState.step >= 5, "TraceState.step should have incremented"


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_hf_trainer_callback_integration():
    """
    Vanilla transformers.Trainer with TraceMLTrainerCallback should emit
    exactly one StepMemoryEvent per optimizer step. The strict equality is
    deliberate: it gates against accidental double-recording from a parallel
    StepMemoryTracker in the callback.
    """
    _reset_traceml_state()
    max_steps = 5

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "results"
        model = _build_tiny_model()
        train_dataset = _TinyTokenizedDataset()
        training_args = _build_training_args(
            str(output_dir), max_steps=max_steps
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[TraceMLTrainerCallback()],
        )
        trainer.train()

        drained = _drain_step_memory_queue(id(model))
        assert len(drained) == max_steps, (
            f"Expected exactly one StepMemoryEvent per optimizer step "
            f"({max_steps}), got {len(drained)}. A count higher than "
            f"max_steps suggests the callback is double-recording memory."
        )

        from traceml_ai.sdk.instrumentation import TraceState

        assert TraceState.step >= max_steps


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_hf_trainer_callback_grad_accum_folds_microbatches():
    """
    With gradient_accumulation_steps=2 and max_steps=3, the callback should
    advance the TraceML step counter exactly 3 times (one TraceML step per
    optimizer step). Each StepTimeBatch should contain ~grad_accum forward
    events, validating that sub-phase auto-timers stay armed across the
    accumulated micro-batches within a single trace_step bracket.
    """
    _reset_traceml_state()
    max_steps = 3
    grad_accum = 2

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "results"
        model = _build_tiny_model()
        train_dataset = _TinyTokenizedDataset(num_rows=40)
        training_args = _build_training_args(
            str(output_dir),
            max_steps=max_steps,
            gradient_accumulation_steps=grad_accum,
        )

        from traceml_ai.runtime.state import get_trace_session_state

        step_before = get_trace_session_state().step

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[TraceMLTrainerCallback()],
        )
        trainer.train()

        step_after = get_trace_session_state().step
        assert step_after - step_before == max_steps, (
            f"Callback should advance step counter by max_steps ({max_steps}), "
            f"not by max_steps * grad_accum. Got delta="
            f"{step_after - step_before}."
        )

        batches = _drain_step_time_queue()
        assert len(batches) == max_steps, (
            f"Expected {max_steps} StepTimeBatches (one per optimizer step), "
            f"got {len(batches)}."
        )

        def _counts(event_name: str) -> list:
            return [
                sum(1 for evt in batch.events if evt.name == event_name)
                for batch in batches
            ]

        forward_counts = _counts("_traceml_internal:forward_time")
        backward_counts = _counts("_traceml_internal:backward_time")
        optimizer_counts = _counts("_traceml_internal:optimizer_step")

        # Each optimizer step calls model.forward() grad_accum times. Each call
        # produces one outermost-forward event. Allow a small tolerance for
        # warmup / HF internals but the dominant count must match.
        assert all(count == grad_accum for count in forward_counts), (
            f"Expected {grad_accum} forward events per TraceML step "
            f"(one per micro-batch in the grad-accum group), got "
            f"{forward_counts}."
        )
        # Backward fires once per micro-batch too, so it folds the same way.
        # Gates against backward auto-timer flags being lost across the
        # accumulated micro-batches.
        assert all(count == grad_accum for count in backward_counts), (
            f"Expected {grad_accum} backward events per TraceML step "
            f"(one per micro-batch in the grad-accum group), got "
            f"{backward_counts}."
        )
        # The optimizer steps exactly once per grad-accum group. Gates against
        # a missing/dummy optimizer event slipping past the forward check.
        assert all(count == 1 for count in optimizer_counts), (
            f"Expected exactly one optimizer event per TraceML step "
            f"(one per optimizer step), got {optimizer_counts}."
        )


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_hf_trainer_wrapper_equivalent_to_direct_callback():
    """
    TraceMLTrainer is now a thin wrapper that auto-installs
    TraceMLTrainerCallback. Running the same tiny setup through both paths
    should produce identical step counts and identical numbers of step-memory
    events. This is the regression gate against the wrapper drifting away
    from direct callback semantics.
    """
    max_steps = 4

    def _run_with(make_trainer) -> tuple:
        _reset_traceml_state()
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "results"
            model = _build_tiny_model()
            train_dataset = _TinyTokenizedDataset()
            training_args = _build_training_args(
                str(output_dir), max_steps=max_steps
            )

            trainer = make_trainer(model, training_args, train_dataset)
            trainer.train()

            from traceml_ai.runtime.state import get_trace_session_state

            step = get_trace_session_state().step
            drained = _drain_step_memory_queue(id(model))
            return step, len(drained)

    callback_steps, callback_samples = _run_with(
        lambda m, a, d: Trainer(
            model=m,
            args=a,
            train_dataset=d,
            callbacks=[TraceMLTrainerCallback()],
        )
    )

    wrapper_steps, wrapper_samples = _run_with(
        lambda m, a, d: TraceMLTrainer(
            model=m,
            args=a,
            train_dataset=d,
            traceml_enabled=True,
        )
    )

    assert callback_steps == max_steps
    assert wrapper_steps == max_steps
    assert callback_samples == max_steps
    assert wrapper_samples == max_steps


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_hf_trainer_wrapper_dedups_user_supplied_callback():
    """
    If a user passes their own TraceMLTrainerCallback in callbacks=[...] and
    also uses TraceMLTrainer, the wrapper must NOT add a second instance.
    Otherwise every step would be double-bracketed and the step counter would
    advance twice per optimizer step.
    """
    _reset_traceml_state()
    max_steps = 3

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "results"
        model = _build_tiny_model()
        train_dataset = _TinyTokenizedDataset()
        training_args = _build_training_args(
            str(output_dir), max_steps=max_steps
        )

        trainer = TraceMLTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[TraceMLTrainerCallback()],
            traceml_enabled=True,
        )

        installed = [
            cb
            for cb in trainer.callback_handler.callbacks
            if isinstance(cb, TraceMLTrainerCallback)
        ]
        assert len(installed) == 1, (
            f"Expected exactly one TraceMLTrainerCallback after dedup, "
            f"found {len(installed)}."
        )

        trainer.train()

        drained = _drain_step_memory_queue(id(model))
        assert len(drained) == max_steps, (
            f"Expected one StepMemoryEvent per optimizer step "
            f"({max_steps}), got {len(drained)}; dedup guard failed."
        )


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_hf_trainer_callback_noop_when_disabled(monkeypatch):
    """
    With TRACEML_DISABLED=1 set after import, the callback must be a complete
    no-op: it advances no step counter and emits no step-memory events. This
    gates the dynamic (per-call) env-var read against regressing to an
    import-time constant that would ignore the kill switch.
    """
    monkeypatch.setenv("TRACEML_DISABLED", "1")
    _reset_traceml_state()
    max_steps = 3

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "results"
        model = _build_tiny_model()
        train_dataset = _TinyTokenizedDataset()
        training_args = _build_training_args(
            str(output_dir), max_steps=max_steps
        )

        from traceml_ai.runtime.state import get_trace_session_state

        step_before = get_trace_session_state().step

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[TraceMLTrainerCallback()],
        )
        trainer.train()

        step_after = get_trace_session_state().step
        assert step_after == step_before, (
            "Step counter must not advance when TRACEML_DISABLED=1; "
            f"advanced by {step_after - step_before}."
        )

        drained = _drain_step_memory_queue(id(model))
        assert (
            drained == []
        ), f"Expected no StepMemoryEvents when disabled, got {len(drained)}."


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_hf_init_enables_dataloader_and_h2d_patches():
    """
    init() must enable the process-wide patches the callback cannot install on
    its own. The DataLoader fetch patch in particular is what lets TraceML
    attribute data-loading time on the HF path; the per-step bracket alone
    never installs it. The H2D Tensor.to patch is gated the same way: the
    auto-timer trace_step arms each step is a no-op unless the patch is on.
    """
    config = init()

    assert config.patch_dataloader, (
        "huggingface.init() must enable DataLoader fetch timing so the "
        "callback path can attribute data-loading time."
    )
    assert config.patch_h2d, (
        "huggingface.init() must enable the H2D Tensor.to patch the "
        "per-step auto-timer relies on."
    )
