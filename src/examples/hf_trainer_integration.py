import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

from traceml.hf_decorators import TraceMLTrainer


def main():
    print("=== TraceMLTrainer E2E Example ===")

    # Configuration
    model_name = "prajjwal1/bert-mini"
    dataset_name = "ag_news"
    batch_size = 32
    num_train_epochs = 3
    # max_steps = 100  # Let epochs control duration

    output_dir = "./hf_trainer_output"
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model & Tokenizer
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=4
    )
    model.to(device)

    # Load Dataset
    print(f"Loading dataset: {dataset_name}")
    raw_dataset = load_dataset(
        dataset_name, split="train[:2000]"
    )  # Larger subset for multi-epoch run

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=64,
        )

    tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)
    # TraceMLTrainer handles column renaming/formatting internally via HFs DataCollator

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        save_strategy="no",
        use_cpu=(device == "cpu"),
        report_to="none",  # Disable wandb etc for this demo
    )

    # Initialize TraceMLTrainer
    print("Initializing TraceMLTrainer with Deep-Dive enabled...")
    trainer = TraceMLTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        traceml_enabled=True,
        traceml_kwargs={
            "trace_layer_memory": True,
            "trace_layer_forward_time": True,
            "trace_layer_backward_time": True,
        },
    )

    # Train
    print("Starting training...")
    trainer.train()
    print("Training complete!")
    print(
        "Check TraceML logs/events (if configured to emit them) or observe console output."
    )


if __name__ == "__main__":
    main()
