import tempfile
from pathlib import Path
from typing import Optional

import pytest
from packaging.version import Version

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("accelerate")

from transformers import (  # noqa: E402
    BertConfig,
    BertForSequenceClassification,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from traceml_ai.integrations.huggingface import (  # noqa: E402
    TraceMLTrainerCallback,
    init,
)


class _TinyTokenizedDataset(torch.utils.data.Dataset):
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
    use_cpu: Optional[bool] = None,
) -> "TrainingArguments":
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_steps=max_steps,
        logging_steps=1,
        use_cpu=not torch.cuda.is_available() if use_cpu is None else use_cpu,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )


def _drain_step_memory_queue() -> list:
    """Drain snapshots from this test's recording session."""
    from traceml_ai.instrumentation.step_events import drain_step_memory_events

    return drain_step_memory_events()


def _drain_step_time_queue() -> list:
    """Drain all StepTimeBatch entries from the shared queue."""
    from traceml_ai.instrumentation.step_events import drain_step_time_batches

    return drain_step_time_batches()


def _reset_traceml_state() -> None:
    """Reset TraceML's process-local step counter and drain shared queues."""
    from traceml_ai.instrumentation.step_events import (
        abort_step_capture,
        begin_step_capture,
    )
    from traceml_ai.runtime.state import reset_trace_session_state

    reset_trace_session_state()
    # Pending events from a previous test would land in our first batch.
    abort_step_capture(begin_step_capture())
    _drain_step_time_queue()
    _drain_step_memory_queue()


def test_hf_trainer_callback_integration():
    """
    Vanilla transformers.Trainer with TraceMLTrainerCallback should emit
    exactly one StepMemoryEvent per optimizer step. The strict equality is
    deliberate: it gates against accidental double-recording from a parallel
    StepMemoryTracker in the callback.
    """
    _reset_traceml_state()
    init()
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

        drained = _drain_step_memory_queue()
        assert len(drained) == max_steps, (
            f"Expected exactly one StepMemoryEvent per optimizer step "
            f"({max_steps}), got {len(drained)}. A count higher than "
            f"max_steps suggests the callback is double-recording memory."
        )

        from traceml_ai.sdk.instrumentation import TraceState

        assert TraceState.step == max_steps


@pytest.mark.parametrize(
    ("grad_accum", "num_rows", "microbatches_per_step"),
    [
        pytest.param(1, 40, [1, 1, 1], id="no-accumulation"),
        pytest.param(2, 40, [2, 2, 2], id="accumulation"),
        pytest.param(4, 40, [4, 4, 2], id="partial-final-group"),
        pytest.param(4, 8, [2], id="epoch-shorter-than-group"),
    ],
)
def test_hf_trainer_callback_grad_accum_folds_microbatches(
    grad_accum, num_rows, microbatches_per_step
):
    """Real HF callback boundaries own counting, including partial groups."""
    _reset_traceml_state()

    # Auto-timers trace_step arms are no-ops until init() installs the patches.
    init()
    max_steps = len(microbatches_per_step)

    from traceml_ai.runtime.state import get_trace_session_state

    # HF starts at zero; TraceML may already have recorded other work.
    trace_state = get_trace_session_state()
    step_before = trace_state.set_step(7)
    boundaries = []

    class BoundaryObserver(TrainerCallback):
        def on_step_begin(self, args, state, control, **kwargs):
            boundaries.append(("begin", state.global_step, trace_state.step))

        def on_substep_end(self, args, state, control, **kwargs):
            boundaries.append(("micro", state.global_step, trace_state.step))

        def on_step_end(self, args, state, control, **kwargs):
            boundaries.append(("end", state.global_step, trace_state.step))

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "results"
        model = _build_tiny_model()
        train_dataset = _TinyTokenizedDataset(num_rows=num_rows)
        training_args = _build_training_args(
            str(output_dir),
            max_steps=max_steps,
            gradient_accumulation_steps=grad_accum,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[TraceMLTrainerCallback(), BoundaryObserver()],
        )
        trainer.train()

        step_after = get_trace_session_state().step
        assert trainer.state.global_step == max_steps
        assert step_after - step_before == max_steps, (
            f"Callback should advance step counter by max_steps ({max_steps}), "
            f"not by max_steps * grad_accum. Got delta="
            f"{step_after - step_before}."
        )

        batches = _drain_step_time_queue()
        assert len(batches) == max_steps, (
            f"Expected {max_steps} StepTimeBatches (one per update boundary), "
            f"got {len(batches)}."
        )
        assert [batch.step for batch in batches] == list(
            range(step_before + 1, step_before + max_steps + 1)
        )
        expected_boundaries = []
        for step, microbatches in enumerate(microbatches_per_step):
            expected_boundaries.append(("begin", step, step_before + step))
            expected_boundaries.extend(
                [("micro", step, step_before + step)] * (microbatches - 1)
            )
            expected_boundaries.append(
                ("end", step + 1, step_before + step + 1)
            )
        assert boundaries == expected_boundaries

        def _counts(event_name: str) -> list:
            return [
                sum(1 for evt in batch.events if evt.name == event_name)
                for batch in batches
            ]

        forward_counts = _counts("_traceml_internal:forward_time")
        backward_counts = _counts("_traceml_internal:backward_time")
        optimizer_counts = _counts("_traceml_internal:optimizer_step")

        # Count actual microbatches, not the configured accumulation maximum.
        assert forward_counts == microbatches_per_step
        assert backward_counts == microbatches_per_step
        # The optimizer steps exactly once per grad-accum group. Gates against
        # a missing/dummy optimizer event slipping past the forward check.
        assert all(count == 1 for count in optimizer_counts), (
            f"Expected exactly one optimizer event per TraceML step "
            f"(one per optimizer step), got {optimizer_counts}."
        )


@pytest.mark.parametrize(
    ("optimizer_name", "optimizer_counts"),
    [
        pytest.param("adamw_torch", [0, 1], id="scaler-skips-call"),
        pytest.param(
            "adamw_torch_fused",
            [1, 1],
            id="fused-call-skips-update",
            marks=pytest.mark.skipif(
                Version(torch.__version__.split("+")[0]) < Version("2.8"),
                reason="Exercise CPU fused AdamW on PyTorch >= 2.8",
            ),
        ),
    ],
)
def test_hf_trainer_callback_counts_update_boundary_when_scaler_skips(
    tmp_path, optimizer_name, optimizer_counts
):
    """Overflow changes optimizer work, not the completed HF step count."""
    _reset_traceml_state()
    init()

    from traceml_ai.runtime.state import get_trace_session_state

    class CPUScaledTrainer(Trainer):
        def create_accelerator_and_postprocess(self):
            super().create_accelerator_and_postprocess()
            # Exercise Accelerate's real scaler/optimizer path without CUDA.
            self.accelerator.scaler = torch.amp.GradScaler("cpu")

        def compute_loss(self, *args, **kwargs):
            loss = super().compute_loss(*args, **kwargs)
            # Overflow all microbatches of the first accumulation group only.
            if self.state.global_step == 0:
                loss = loss * float("inf")
            return loss

    model = _build_tiny_model()
    initial_parameters = [p.detach().clone() for p in model.parameters()]
    observations = []
    scales = []

    class UpdateObserver(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            parameters_unchanged = all(
                torch.equal(initial, current)
                for initial, current in zip(
                    initial_parameters, model.parameters()
                )
            )
            observations.append(
                (
                    state.global_step,
                    get_trace_session_state().step,
                    parameters_unchanged,
                )
            )
            scales.append(trainer.accelerator.scaler.get_scale())

    args = _build_training_args(
        str(tmp_path), max_steps=2, gradient_accumulation_steps=2, use_cpu=True
    )
    args.max_grad_norm = 0.0
    args.optim = optimizer_name
    trainer = CPUScaledTrainer(
        model=model,
        args=args,
        train_dataset=_TinyTokenizedDataset(),
        callbacks=[TraceMLTrainerCallback(), UpdateObserver()],
    )
    initial_scale = trainer.accelerator.scaler.get_scale()
    trainer.train()

    assert observations == [(1, 1, True), (2, 2, False)]
    assert scales == [initial_scale / 2, initial_scale / 2]
    batches = _drain_step_time_queue()
    assert [batch.step for batch in batches] == [1, 2]
    for name, expected in (
        ("_traceml_internal:forward_time", [2, 2]),
        ("_traceml_internal:backward_time", [2, 2]),
        ("_traceml_internal:optimizer_step", optimizer_counts),
    ):
        assert [
            sum(evt.name == name for evt in batch.events) for batch in batches
        ] == expected


def test_hf_trainer_optional_callback_preserves_training():
    """Conditional callback registration preserves losses and parameters."""
    max_steps = 4

    def _run_with(enable_tracing: bool) -> tuple:
        _reset_traceml_state()
        torch.manual_seed(42)
        callbacks = []
        if enable_tracing:
            init()
            callbacks.append(TraceMLTrainerCallback())

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "results"
            model = _build_tiny_model()
            train_dataset = _TinyTokenizedDataset()
            training_args = _build_training_args(
                str(output_dir), max_steps=max_steps, use_cpu=True
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                callbacks=callbacks,
            )
            result = trainer.train()
            assert trainer.state.global_step == max_steps

            from traceml_ai.runtime.state import get_trace_session_state

            step = get_trace_session_state().step
            drained = _drain_step_memory_queue()
            batches = _drain_step_time_queue()
            parameters = {
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            }
            return (
                result.training_loss,
                parameters,
                step,
                len(drained),
                len(batches),
            )

    untraced = _run_with(False)
    traced = _run_with(True)
    assert untraced[2:] == (0, 0, 0)
    assert traced[2:] == (max_steps, max_steps, max_steps)
    assert traced[0] == pytest.approx(untraced[0])
    torch.testing.assert_close(traced[1], untraced[1])


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

        drained = _drain_step_memory_queue()
        assert (
            drained == []
        ), f"Expected no StepMemoryEvents when disabled, got {len(drained)}."


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


# --- TraceML telemetry-completeness guard (instrumentation hardening) -------
# test_hf_init_enables_dataloader_and_h2d_patches proves init() REQUESTS the
# DataLoader patch (config flags). This guard goes one step further and
# proves the stream actually EMITS during a real Trainer run. Absences (a
# stream silently dark) are the costly failure mode; a config flag cannot
# catch a broken patch, a renamed event, or a capture that never completes.

DATALOADER_STREAM = "_traceml_internal:dataloader_next"


def test_hf_callback_run_emits_dataloader_fetch_events():
    """
    COMPLETENESS guard: with huggingface.init() called, a vanilla Trainer +
    TraceMLTrainerCallback run must land `_traceml_internal:dataloader_next`
    TimeEvents in completed StepTimeBatches, not merely set the
    patch_dataloader config flag. Gates the full path: patch install ->
    fetch timing -> step capture -> per-step completion.
    """
    _reset_traceml_state()

    # Drop STEP-scope events still sitting in the pending capture from
    # earlier tests; they would otherwise fold into this run's first batch
    # and could fake a pass.
    from traceml_ai.instrumentation.step_events import (
        abort_step_capture,
        begin_step_capture,
    )

    abort_step_capture(begin_step_capture())

    # Explicit init is the documented HF path; the DataLoader fetch patch
    # is process-wide and only installed here, never by the callback.
    # init() is idempotent for the same effective config, so this is safe
    # whether or not the patch-flag test already ran.
    init()

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

    batches = _drain_step_time_queue()
    names = sorted({evt.name for batch in batches for evt in batch.events})
    assert DATALOADER_STREAM in names, (
        "DataLoader-fetch telemetry is DARK: the HF run did not emit "
        f"'{DATALOADER_STREAM}'. Streams seen: {names}"
    )
