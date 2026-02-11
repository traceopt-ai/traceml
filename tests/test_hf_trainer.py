import math
import shutil
import tempfile
from pathlib import Path

import pytest

from traceml.hf_decorators import TraceMLTrainer

try:
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_hf_trainer_integration():
    """
    Test that TraceMLTrainer runs a few steps with a real model and
    TraceML instrumentation enabled.
    """
    # Use a tiny BERT model for speed
    model_name = "prajjwal1/bert-mini"

    # Create temp directory for outputs
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "results"

        # Load model and tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=4
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load tiny dataset
        dataset = load_dataset("ag_news", split="train[:20]")

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=32,
            )

        tokenized_datasets = dataset.map(tokenize_function, batched=True)

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
            train_dataset=tokenized_datasets,
            traceml_enabled=True,
        )

        # Capture stdout/stderr/logs Or just check if it runs without error.
        # For now, just ensuring it completes without error is the primary goal.
        trainer.train()

        # Verification: Check if TraceState.step incremented
        # We can import TraceState to check global state
        from traceml.decorators import TraceState

        # Note: TraceState is global, so it might be > 5 if other tests ran.
        # But we know it should be at least 5 more than before?
        # Since tests run in isolation or we can reset.
        assert TraceState.step >= 5, "TraceState.step should have incremented"
