import os

import torch
from datasets import load_dataset
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomResizedCrop,
    ToTensor,
)
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    DefaultDataCollator,
    TrainingArguments,
)

from traceml.hf_decorators import TraceMLTrainer


def main():
    print("=== TraceMLTrainer Vision Example (ViT) ===")

    # Configuration
    model_name = "google/vit-base-patch16-224-in21k"
    dataset_name = "cifar10"
    batch_size = 16  # ViT is heavier
    num_train_epochs = 1

    output_dir = "./hf_trainer_vision_output"
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load Model & Processor
    print(f"Loading model: {model_name}")
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=10,
        id2label={i: str(i) for i in range(10)},
        label2id={str(i): i for i in range(10)},
    )
    model.to(device)

    # Load Dataset
    print(f"Loading dataset: {dataset_name}")
    # Load tiny subset
    dataset = load_dataset(dataset_name, split="train[:500]")

    # Transform
    normalize = Normalize(
        mean=image_processor.image_mean, std=image_processor.image_std
    )
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    def transforms(examples):
        examples["pixel_values"] = [
            _transforms(img.convert("RGB")) for img in examples["img"]
        ]
        del examples["img"]
        return examples

    dataset = dataset.with_transform(transforms)

    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        logging_steps=5,
        save_strategy="no",
        use_cpu=(device == "cpu"),
        report_to="none",
        disable_tqdm=True,
        remove_unused_columns=False,  # Required for vision datasets sometimes
    )

    # Initialize TraceMLTrainer
    print("Initializing TraceMLTrainer with Deep-Dive...")
    trainer = TraceMLTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DefaultDataCollator(),
        traceml_enabled=True,
        traceml_kwargs={
            "sample_layer_memory": True,
            "trace_layer_forward_memory": True,
            "trace_layer_forward_time": True,
            "trace_layer_backward_time": True,
            # Explicitly include layer names to be sure?
            # "include_names": ["vit.encoder.layer"],
        },
    )

    # Train
    print("Starting training...")
    trainer.train()
    print("Training complete!")


if __name__ == "__main__":
    main()
