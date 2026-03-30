"""Single-pass AST visitor for extracting ML training-script patterns."""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Set

from traceml.utils.ast_analysis.helpers import (
    extract_assigned_name,
    extract_autocast_dtype,
    fqn,
    is_cudnn_attr,
    is_in_training_loop,
    kw_const,
    kw_repr,
    location,
    phase_hint_for_node,
    safe_float,
    safe_int,
)
from traceml.utils.ast_analysis.models import (
    DataLoaderFinding,
    DistributedFinding,
    FineTuningFinding,
    HFTrainingArgumentsFinding,
    ModelFinding,
    OptimizerFinding,
    PrecisionFinding,
    ScriptLocation,
)

_DATALOADER_SUFFIXES = {"DataLoader", "dataloader.DataLoader"}

_KNOWN_OPTIMIZERS: Dict[str, str] = {
    "torch.optim.Adam": "Adam",
    "torch.optim.AdamW": "AdamW",
    "torch.optim.SGD": "SGD",
    "torch.optim.RMSprop": "RMSprop",
    "torch.optim.Adagrad": "Adagrad",
    "torch.optim.LBFGS": "LBFGS",
    "bitsandbytes.optim.Adam8bit": "Adam8bit",
    "bitsandbytes.optim.AdamW8bit": "AdamW8bit",
    "apex.optimizers.FusedAdam": "FusedAdam",
    "apex.optimizers.FusedLAMB": "FusedLAMB",
    "deepspeed.ops.adam.FusedAdam": "FusedAdam",
    "deepspeed.ops.adam.DeepSpeedCPUAdam": "DeepSpeedCPUAdam",
    "lion_pytorch.Lion": "Lion",
    "transformers.optimization.Adafactor": "Adafactor",
}

_PEFT_METHODS: Dict[str, str] = {
    "LoraConfig": "lora",
    "peft.LoraConfig": "lora",
    "AdaLoraConfig": "lora",
    "peft.AdaLoraConfig": "lora",
    "get_peft_model": "peft_generic",
    "peft.get_peft_model": "peft_generic",
    "IA3Config": "peft_generic",
    "peft.IA3Config": "peft_generic",
    "PrefixTuningConfig": "adapter",
    "peft.PrefixTuningConfig": "adapter",
    "PromptTuningConfig": "adapter",
    "peft.PromptTuningConfig": "adapter",
}

_QLORA_SIGNALS = {"prepare_model_for_kbit_training", "BitsAndBytesConfig"}

_HF_TRAINING_ARG_CLASSES = {
    "TrainingArguments",
    "Seq2SeqTrainingArguments",
    "transformers.TrainingArguments",
    "transformers.Seq2SeqTrainingArguments",
}

_HF_TRAINER_CLASSES = {
    "Trainer",
    "Seq2SeqTrainer",
    "transformers.Trainer",
    "transformers.Seq2SeqTrainer",
    "trl.SFTTrainer",
    "trl.DPOTrainer",
    "trl.PPOTrainer",
    "trl.RewardTrainer",
    "SFTTrainer",
    "DPOTrainer",
    "PPOTrainer",
    "RewardTrainer",
}

_LIGHTNING_PREFIXES = ("pytorch_lightning", "lightning.pytorch", "lightning")


class PatternVisitor(ast.NodeVisitor):
    """Accumulate relevant findings in a single AST traversal."""

    def __init__(
        self,
        script_path: str,
        lines: List[str],
        imports: Dict[str, str],
        str_consts: Dict[str, str],
        parent_map: Optional[Dict[int, ast.AST]] = None,
    ) -> None:
        self._path = script_path
        self._lines = lines
        self._imports = imports
        self._str_consts = str_consts
        self._parent_map: Dict[int, ast.AST] = parent_map or {}

        self.dataloaders: List[DataLoaderFinding] = []
        self.optimizers: List[OptimizerFinding] = []
        self.precision: List[PrecisionFinding] = []
        self.distributed: List[DistributedFinding] = []
        self.models: List[ModelFinding] = []
        self.fine_tuning: List[FineTuningFinding] = []
        self.hf_training_args: List[HFTrainingArgumentsFinding] = []
        self.gradient_accumulation_steps: Optional[int] = None

        self.phase_hints: Set[str] = set()
        self.trainer_train_called: bool = False
        self._seen_lightning = False

        self.sync_calls_item: int = 0
        self.sync_calls_cpu: int = 0
        self.sync_calls_numpy: int = 0
        self.sync_calls_cuda_synchronize: int = 0

        self.to_device_detected: bool = False
        self.non_blocking_used: bool = False

        self.zero_grad_detected: bool = False
        self.backward_detected: bool = False
        self.optimizer_step_detected: bool = False
        self.logging_in_loop: bool = False
        self.checkpoint_in_loop: bool = False
        self.validation_in_loop: bool = False

        self.distributed_sampler_used: bool = False
        self.set_epoch_called: bool = False
        self.prefetch_factor_set: bool = False

    def _loc(self, node: ast.AST) -> ScriptLocation:
        return location(self._path, node, self._lines)

    def _resolve(self, call: ast.Call) -> Optional[str]:
        return fqn(call, self._imports)

    def _phase(self, node: ast.AST) -> Optional[str]:
        phase = phase_hint_for_node(node, self._parent_map)
        if phase:
            self.phase_hints.add(phase)
        return phase

    def _var_name(self, node: ast.AST) -> Optional[str]:
        return extract_assigned_name(node, self._parent_map)

    def _add_dist(
        self,
        node: ast.AST,
        kind: str,
        backend: Optional[str] = None,
    ) -> None:
        self.distributed.append(
            DistributedFinding(
                location=self._loc(node),
                kind=kind,
                backend=backend,
            )
        )

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = self._resolve(node)
        if name is not None:
            self._check_dataloader(node, name)
            self._check_optimizer(node, name)
            self._check_precision_call(node, name)
            self._check_distributed(node, name)
            self._check_model(node, name)
            self._check_peft(node, name)
            self._check_hf_training_arguments(node, name)
            self._check_sync_calls(node, name)
            self._check_device_transfer(node, name)
            self._check_train_loop_flags(node, name)
            self._check_distributed_sampler(node, name)
            self._check_set_epoch(node, name)
            self._check_wrapper_calls(node, name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr == "benchmark"
            ):
                if is_cudnn_attr(target):
                    self.models.append(
                        ModelFinding(
                            location=self._loc(node),
                            kind="cudnn_benchmark",
                            model_name=None,
                        )
                    )
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        if not self._seen_lightning:
            for base in node.bases:
                bname = (
                    base.id
                    if isinstance(base, ast.Name)
                    else base.attr if isinstance(base, ast.Attribute) else None
                )
                if bname == "LightningModule":
                    src = self._imports.get("LightningModule", "")
                    if not src or any(
                        src.startswith(p) for p in _LIGHTNING_PREFIXES
                    ):
                        self._seen_lightning = True
                        self._add_dist(node, "lightning")
                        break
        self.generic_visit(node)

    def _check_dataloader(self, node: ast.Call, name: str) -> None:
        if not (name.endswith(".DataLoader") or name in _DATALOADER_SUFFIXES):
            return

        raw: Dict[str, str] = {}
        for kw_name in (
            "batch_size",
            "num_workers",
            "pin_memory",
            "persistent_workers",
            "prefetch_factor",
        ):
            v = kw_repr(node, kw_name)
            if v is not None:
                raw[kw_name] = v

        bs = safe_int(kw_const(node, "batch_size"))
        nw = safe_int(kw_const(node, "num_workers"))
        pm_raw = kw_const(node, "pin_memory")
        pm = bool(pm_raw) if pm_raw is not None else None
        pw_raw = kw_const(node, "persistent_workers")
        pw = bool(pw_raw) if pw_raw is not None else None
        pf = safe_int(kw_const(node, "prefetch_factor"))

        if pf is not None:
            self.prefetch_factor_set = True

        self.dataloaders.append(
            DataLoaderFinding(
                location=self._loc(node),
                batch_size=bs,
                num_workers=nw,
                pin_memory=pm,
                persistent_workers=pw,
                prefetch_factor=pf,
                raw_kwargs=raw,
                variable_name=self._var_name(node),
                phase_hint=self._phase(node),
            )
        )

    def _check_optimizer(self, node: ast.Call, name: str) -> None:
        opt_type: Optional[str] = None
        for fqn_key, label in _KNOWN_OPTIMIZERS.items():
            leaf = fqn_key.rsplit(".", 1)[-1]
            if name == fqn_key or name == leaf or name.endswith(f".{leaf}"):
                opt_type = label
                break
        if opt_type is None:
            return

        self.optimizers.append(
            OptimizerFinding(
                location=self._loc(node),
                optimizer_type=opt_type,
                learning_rate=safe_float(kw_const(node, "lr")),
                weight_decay=safe_float(kw_const(node, "weight_decay")),
                variable_name=self._var_name(node),
                phase_hint=self._phase(node),
            )
        )

    def _check_precision_call(self, node: ast.Call, name: str) -> None:
        phase = self._phase(node)

        if "autocast" in name:
            dtype_str = extract_autocast_dtype(node)
            self.precision.append(
                PrecisionFinding(
                    location=self._loc(node),
                    kind="autocast",
                    dtype_str=dtype_str,
                    is_deprecated_api="cuda.amp.autocast" in name,
                    phase_hint=phase,
                )
            )
            return

        if "GradScaler" in name:
            self.precision.append(
                PrecisionFinding(
                    location=self._loc(node),
                    kind="grad_scaler",
                    dtype_str=None,
                    is_deprecated_api="cuda.amp.GradScaler" in name,
                    phase_hint=phase,
                )
            )
            return

        if isinstance(node.func, ast.Attribute):
            meth = node.func.attr
            if meth == "half":
                self.precision.append(
                    PrecisionFinding(
                        location=self._loc(node),
                        kind="half",
                        dtype_str="float16",
                        is_deprecated_api=False,
                        phase_hint=phase,
                    )
                )
            elif meth == "bfloat16":
                self.precision.append(
                    PrecisionFinding(
                        location=self._loc(node),
                        kind="bfloat16",
                        dtype_str="bfloat16",
                        is_deprecated_api=False,
                        phase_hint=phase,
                    )
                )

    def _check_distributed(self, node: ast.Call, name: str) -> None:
        if "DistributedDataParallel" in name or name.endswith(".DDP"):
            self._add_dist(node, "ddp")
            return

        if "FullyShardedDataParallel" in name or name.endswith(".FSDP"):
            self._add_dist(node, "fsdp")
            return

        if "init_process_group" in name:
            backend: Optional[str] = None
            if node.args and isinstance(node.args[0], ast.Constant):
                backend = str(node.args[0].value)
            else:
                raw = kw_const(node, "backend")
                if isinstance(raw, str):
                    backend = raw
            self._add_dist(node, "init_process_group", backend)
            return

        if (
            name == "torch.compile" or name.endswith(".compile")
        ) and "torch" in name:
            self._add_dist(node, "torch_compile")
            return

        if (
            "Accelerator" in name
            and "accelerate"
            in self._imports.get(
                name.split(".")[0],
                name,
            ).lower()
        ):
            self._add_dist(node, "accelerate")
            return

        if not self._seen_lightning:
            if (
                any(name.startswith(p) for p in _LIGHTNING_PREFIXES)
                and "Trainer" in name
            ):
                self._seen_lightning = True
                self._add_dist(node, "lightning")
                return
            if name == "Trainer" or name.endswith(".Trainer"):
                src = self._imports.get("Trainer", "")
                if any(src.startswith(p) for p in _LIGHTNING_PREFIXES):
                    self._seen_lightning = True
                    self._add_dist(node, "lightning")
                    return

    def _check_model(self, node: ast.Call, name: str) -> None:
        if "from_pretrained" in name:
            model_id: Optional[str] = None
            if node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(
                    arg0.value, str
                ):
                    model_id = arg0.value
                elif (
                    isinstance(arg0, ast.Name) and arg0.id in self._str_consts
                ):
                    model_id = self._str_consts[arg0.id]
            self.models.append(
                ModelFinding(
                    location=self._loc(node),
                    kind="from_pretrained",
                    model_name=model_id,
                )
            )
            return

        if "gradient_checkpointing_enable" in name:
            self.models.append(
                ModelFinding(
                    location=self._loc(node),
                    kind="gradient_checkpointing",
                    model_name=None,
                )
            )
            return

        if "set_float32_matmul_precision" in name:
            self.models.append(
                ModelFinding(
                    location=self._loc(node),
                    kind="float32_matmul_precision",
                    model_name=None,
                )
            )

    def _check_peft(self, node: ast.Call, name: str) -> None:
        for key, method in _PEFT_METHODS.items():
            leaf = key.rsplit(".", 1)[-1]
            if name == key or name == leaf or name.endswith(f".{leaf}"):
                details: Dict[str, str] = {}
                r = kw_const(node, "r")
                if r is not None:
                    details["rank"] = str(r)
                alpha = kw_const(node, "lora_alpha")
                if alpha is not None:
                    details["lora_alpha"] = str(alpha)
                self.fine_tuning.append(
                    FineTuningFinding(
                        location=self._loc(node),
                        method=method,
                        details=details,
                    )
                )
                return

        leaf = name.rsplit(".", 1)[-1]
        if leaf in _QLORA_SIGNALS or name in _QLORA_SIGNALS:
            pass

    def _check_hf_training_arguments(self, node: ast.Call, name: str) -> None:
        leaf = name.rsplit(".", 1)[-1]
        if name not in _HF_TRAINING_ARG_CLASSES and leaf not in {
            n.rsplit(".", 1)[-1] for n in _HF_TRAINING_ARG_CLASSES
        }:
            return

        self.hf_training_args.append(
            HFTrainingArgumentsFinding(
                location=self._loc(node),
                fp16=kw_const(node, "fp16"),
                bf16=kw_const(node, "bf16"),
                per_device_train_batch_size=safe_int(
                    kw_const(node, "per_device_train_batch_size")
                ),
                per_device_eval_batch_size=safe_int(
                    kw_const(node, "per_device_eval_batch_size")
                ),
                gradient_accumulation_steps=safe_int(
                    kw_const(node, "gradient_accumulation_steps")
                ),
                dataloader_num_workers=safe_int(
                    kw_const(node, "dataloader_num_workers")
                ),
                dataloader_pin_memory=kw_const(node, "dataloader_pin_memory"),
                dataloader_persistent_workers=kw_const(
                    node, "dataloader_persistent_workers"
                ),
                learning_rate=safe_float(kw_const(node, "learning_rate")),
                weight_decay=safe_float(kw_const(node, "weight_decay")),
                optim=kw_const(node, "optim"),
                logging_steps=safe_int(kw_const(node, "logging_steps")),
                save_steps=safe_int(kw_const(node, "save_steps")),
                eval_steps=safe_int(kw_const(node, "eval_steps")),
                max_steps=safe_int(kw_const(node, "max_steps")),
                num_train_epochs=safe_float(
                    kw_const(node, "num_train_epochs")
                ),
                gradient_checkpointing=kw_const(
                    node, "gradient_checkpointing"
                ),
                torch_compile=kw_const(node, "torch_compile"),
            )
        )

    def _check_sync_calls(self, node: ast.Call, name: str) -> None:
        if not isinstance(node.func, ast.Attribute):
            return
        attr = node.func.attr
        in_loop = is_in_training_loop(node, self._parent_map)

        if attr == "item" and in_loop:
            self.sync_calls_item += 1
        elif attr == "cpu" and in_loop:
            self.sync_calls_cpu += 1
        elif attr == "numpy" and in_loop:
            self.sync_calls_numpy += 1
        elif (
            "cuda" in name and "synchronize" in name
        ) or attr == "synchronize":
            self.sync_calls_cuda_synchronize += 1

    def _check_device_transfer(self, node: ast.Call, name: str) -> None:
        if not isinstance(node.func, ast.Attribute):
            return
        if node.func.attr not in ("to", "cuda"):
            return
        self.to_device_detected = True
        if kw_const(node, "non_blocking") is True:
            self.non_blocking_used = True

    def _check_train_loop_flags(self, node: ast.Call, name: str) -> None:
        in_loop = is_in_training_loop(node, self._parent_map)

        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr == "zero_grad":
                self.zero_grad_detected = True
            elif attr == "backward":
                self.backward_detected = True
            elif attr == "step":
                self.optimizer_step_detected = True
            elif (
                attr in ("save", "save_checkpoint", "save_pretrained")
                and in_loop
            ):
                self.checkpoint_in_loop = True
            elif attr == "eval" and in_loop:
                self.validation_in_loop = True
            elif attr == "no_grad" and in_loop:
                self.validation_in_loop = True
            elif attr in ("log", "add_scalar") and in_loop:
                self.logging_in_loop = True
        elif isinstance(node.func, ast.Name):
            fn = node.func.id
            if fn in ("no_grad", "inference_mode") and in_loop:
                self.validation_in_loop = True
            elif fn == "print" and in_loop:
                self.logging_in_loop = True

    def _check_distributed_sampler(self, node: ast.Call, name: str) -> None:
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "DistributedSampler":
            self.distributed_sampler_used = True

    def _check_set_epoch(self, node: ast.Call, name: str) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "set_epoch"
        ):
            self.set_epoch_called = True

    def _check_wrapper_calls(self, node: ast.Call, name: str) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "train":
            self.trainer_train_called = True
            self.phase_hints.add("train")

        leaf = name.rsplit(".", 1)[-1]
        if name in _HF_TRAINER_CLASSES or leaf in {
            n.rsplit(".", 1)[-1] for n in _HF_TRAINER_CLASSES
        }:
            if not any(d.kind == "hf_trainer" for d in self.distributed):
                self._add_dist(node, "hf_trainer")
