import tempfile
from pathlib import Path

import pytest

from traceml.integrations.huggingface import TraceMLTrainer

try:
    import torch
    from transformers import (
        BertConfig,
        BertForSequenceClassification,
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


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_hf_trainer_integration():
    """
    Test that TraceMLTrainer runs a few steps with a real model and
    TraceML instrumentation enabled.
    """
    # Create temp directory for outputs
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "results"

        # Build a very small local BERT model so the test does not depend on
        # Hub availability or third-party model metadata.
        model = BertForSequenceClassification(
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
        train_dataset = _TinyTokenizedDataset()

        training_args = TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=4,
            max_steps=5,  # Run only 5 steps
            logging_steps=1,
            use_cpu=not torch.cuda.is_available(),
            save_strategy="no",
        )

        trainer = TraceMLTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            traceml_enabled=True,
        )

        # Capture stdout/stderr/logs Or just check if it runs without error.
        # For now, just ensuring it completes without error is the primary goal.
        trainer.train()

        # Verification: Check if TraceState.step incremented
        # We can import TraceState to check global state
        from traceml.sdk.decorators_compat import TraceState

        # Note: TraceState is global, so it might be > 5 if other tests ran.
        # But we know it should be at least 5 more than before?
        # Since tests run in isolation or we can reset.
        assert TraceState.step >= 5, "TraceState.step should have incremented"
