"""Public result types for AST-based training-script analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class ScriptLocation:
    """Exact source position of a finding."""

    file_path: str
    line: int
    col: int
    text: str


@dataclass
class DataLoaderFinding:
    """Static information about a DataLoader-like constructor call."""

    location: ScriptLocation
    batch_size: Optional[int]
    num_workers: Optional[int]
    pin_memory: Optional[bool]
    persistent_workers: Optional[bool]
    prefetch_factor: Optional[int]
    raw_kwargs: Dict[str, str]
    variable_name: Optional[str] = None
    phase_hint: Optional[str] = None


@dataclass
class OptimizerFinding:
    """Static information about an optimizer constructor call."""

    location: ScriptLocation
    optimizer_type: str
    learning_rate: Optional[float]
    weight_decay: Optional[float]
    variable_name: Optional[str] = None
    phase_hint: Optional[str] = None


@dataclass
class PrecisionFinding:
    """Static signal for precision-related constructs."""

    location: ScriptLocation
    kind: str  # "autocast" | "grad_scaler" | "half" | "bfloat16"
    dtype_str: Optional[str]
    is_deprecated_api: bool
    phase_hint: Optional[str] = None


@dataclass
class DistributedFinding:
    """Static signal for distributed or wrapper strategy usage."""

    location: ScriptLocation
    kind: (
        str  # "ddp" | "fsdp" | "accelerate" | "lightning" | "hf_trainer" | ...
    )
    backend: Optional[str]


@dataclass
class ModelFinding:
    """Static signal for model provenance or performance-relevant model config."""

    location: ScriptLocation
    kind: str  # "from_pretrained" | "gradient_checkpointing" | ...
    model_name: Optional[str]


@dataclass
class FineTuningFinding:
    """Static signal for adapter / PEFT style fine-tuning."""

    location: ScriptLocation
    method: str  # "lora" | "qlora" | "peft_generic" | "adapter"
    details: Dict[str, str] = field(default_factory=dict)


@dataclass
class HFTrainingArgumentsFinding:
    """Relevant subset of Hugging Face TrainingArguments fields."""

    location: ScriptLocation
    fp16: Optional[bool] = None
    bf16: Optional[bool] = None
    per_device_train_batch_size: Optional[int] = None
    per_device_eval_batch_size: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = None
    dataloader_num_workers: Optional[int] = None
    dataloader_pin_memory: Optional[bool] = None
    dataloader_persistent_workers: Optional[bool] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    optim: Optional[str] = None
    logging_steps: Optional[int] = None
    save_steps: Optional[int] = None
    eval_steps: Optional[int] = None
    max_steps: Optional[int] = None
    num_train_epochs: Optional[float] = None
    gradient_checkpointing: Optional[bool] = None
    torch_compile: Optional[bool] = None


@dataclass
class CodeFindings:
    """All ML patterns found in a single script.

    Public entry points always return this object, even when analysis fails.
    Failures are recorded in ``parse_errors``.
    """

    script_path: str
    dataloaders: List[DataLoaderFinding] = field(default_factory=list)
    optimizers: List[OptimizerFinding] = field(default_factory=list)
    precision: List[PrecisionFinding] = field(default_factory=list)
    distributed: List[DistributedFinding] = field(default_factory=list)
    models: List[ModelFinding] = field(default_factory=list)
    fine_tuning: List[FineTuningFinding] = field(default_factory=list)
    hf_training_args: List[HFTrainingArgumentsFinding] = field(
        default_factory=list
    )

    imports: Dict[str, str] = field(default_factory=dict)
    has_training_loop: bool = False
    has_gradient_checkpointing: bool = False
    gradient_accumulation_steps: Optional[int] = None
    parse_errors: List[str] = field(default_factory=list)

    phase_hints: Set[str] = field(default_factory=set)
    trainer_train_called: bool = False

    sync_calls_item: int = 0
    sync_calls_cpu: int = 0
    sync_calls_numpy: int = 0
    sync_calls_cuda_synchronize: int = 0

    to_device_detected: bool = False
    non_blocking_used: bool = False

    zero_grad_detected: bool = False
    backward_detected: bool = False
    optimizer_step_detected: bool = False
    logging_in_loop: bool = False
    checkpoint_in_loop: bool = False
    validation_in_loop: bool = False

    distributed_sampler_used: bool = False
    set_epoch_called: bool = False
    prefetch_factor_set: bool = False
