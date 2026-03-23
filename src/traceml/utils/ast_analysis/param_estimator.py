"""Static parameter estimator — walks AST to count model parameters.

Two strategies, applied in order of confidence:

1.  **Registry Lookup** (highest confidence)
    If the script calls `from_pretrained("bert-base-uncased")` and the name
    is found in the model registry, use the exact known parameter count.

2.  **Layer Counting** (medium confidence)
    Walk the AST for common PyTorch layer constructors and accumulate params:
      - nn.Linear(in, out, bias=True)   → in*out + out (if bias)
      - nn.Embedding(num, dim)           → num*dim
      - nn.Conv2d(c_in, c_out, k)       → c_in*c_out*k*k + c_out (if bias)
      - nn.Conv1d(c_in, c_out, k)       → c_in*c_out*k + c_out (if bias)
      - nn.ConvTranspose2d              → same as Conv2d
      - nn.LayerNorm(shape)             → 2*shape
      - nn.BatchNorm1d/2d/3d(num)       → 2*num
      - nn.MultiheadAttention(d, h)     → 4*d*d + 4*d

Never crashes — returns 0 on any failure.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Optional

from traceml.utils.ast_analysis.model_registry import ModelSpec, lookup_model
from traceml.utils.ast_scanner import CodeFindings


@dataclass
class ParamEstimate:
    num_params: int
    bytes_per_param: int  # 4=fp32, 2=fp16/bf16
    source: str  # "registry", "layer_count", "unknown"
    model_description: str = ""

    @property
    def total_bytes(self) -> int:
        return self.num_params * self.bytes_per_param

    @property
    def total_gb(self) -> float:
        return self.total_bytes / (1024**3)


# ---------------------------------------------------------------------------
# Precision resolution
# ---------------------------------------------------------------------------


def _resolve_bytes_per_param(findings: CodeFindings) -> int:
    """Infer bytes-per-param from precision findings.

    Returns 2 for fp16/bf16, 4 for fp32 (default).
    """
    for p in findings.precision:
        if p.dtype_str in ("float16", "bfloat16", "half"):
            return 2
        if p.kind in ("half", "bfloat16"):
            return 2
    return 4


# ---------------------------------------------------------------------------
# Registry-based estimation (LLMs / HuggingFace)
# ---------------------------------------------------------------------------


def _estimate_from_registry(
    findings: CodeFindings, bytes_per_param: int
) -> Optional[ParamEstimate]:
    """Try to match any from_pretrained() call to the model registry."""
    for model_finding in findings.models:
        if (
            model_finding.kind == "from_pretrained"
            and model_finding.model_name
        ):
            spec: Optional[ModelSpec] = lookup_model(model_finding.model_name)
            if spec:
                # Honour precision from script if lower than default
                bpp = min(bytes_per_param, spec.bytes_per_param)
                return ParamEstimate(
                    num_params=spec.num_params,
                    bytes_per_param=bpp,
                    source="registry",
                    model_description=spec.description
                    or model_finding.model_name,
                )
    return None


# ---------------------------------------------------------------------------
# AST layer counting (custom DL models)
# ---------------------------------------------------------------------------


def _collect_module_constants(tree: ast.AST) -> dict:
    """Scan top-level assignments of the form NAME = <integer or arithmetic>.

    Handles common patterns:
        INPUT_DIM = 1024
        HIDDEN = INPUT_DIM * 2   (if INPUT_DIM already resolved)

    Returns a dict of {name: int_value}.
    """
    consts: dict = {}
    # Two passes: first literals, then expressions that reference prior names.
    for _pass in range(2):
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            # Only simple name targets at module level
            if len(node.targets) != 1 or not isinstance(
                node.targets[0], ast.Name
            ):
                continue
            name = node.targets[0].id
            if name in consts:
                continue
            val = _eval_const_expr(node.value, consts)
            if isinstance(val, int):
                consts[name] = val
    return consts


def _eval_const_expr(node: ast.AST, consts: dict):
    """Recursively evaluate a simple constant arithmetic AST expression.

    Handles: int literals, Name lookups, BinOp (+, -, *, //, /).
    Returns int or None.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.Name) and node.id in consts:
        return consts[node.id]
    if isinstance(node, ast.BinOp):
        left = _eval_const_expr(node.left, consts)
        right = _eval_const_expr(node.right, consts)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, (ast.Div, ast.FloorDiv)):
                return left // right
        except Exception:
            return None
    return None


def _get_int_arg(
    node: ast.Call, pos: int, name: str, consts: dict
) -> int | None:
    """Get an integer from a positional or keyword arg, resolving named constants."""
    # positional
    if pos < len(node.args):
        a = node.args[pos]
        val = _eval_const_expr(a, consts)
        if isinstance(val, int):
            return val
    # keyword
    for kw in node.keywords:
        if kw.arg == name:
            val = _eval_const_expr(kw.value, consts)
            if isinstance(val, int):
                return val
    return None


def _get_bool_kwarg(node: ast.Call, name: str, default: bool = True) -> bool:
    for kw in node.keywords:
        if kw.arg == name:
            if isinstance(kw.value, ast.Constant):
                return bool(kw.value.value)
            if isinstance(kw.value, ast.NameConstant):  # Python 3.7
                return bool(kw.value.value)
    return default


def _kernel_size(node: ast.Call, consts: dict, pos: int = 2) -> int:
    """Return kernel_size as int (handles single int, named const, or tuple)."""
    if pos < len(node.args):
        a = node.args[pos]
        val = _eval_const_expr(a, consts)
        if isinstance(val, int):
            return val
        if isinstance(a, ast.Tuple) and a.elts:
            val = _eval_const_expr(a.elts[0], consts)
            if isinstance(val, int):
                return val
    for kw in node.keywords:
        if kw.arg == "kernel_size":
            val = _eval_const_expr(kw.value, consts)
            if isinstance(val, int):
                return val
            if isinstance(kw.value, ast.Tuple) and kw.value.elts:
                val = _eval_const_expr(kw.value.elts[0], consts)
                if isinstance(val, int):
                    return val
    return 3  # safe fallback


def _count_params_from_layers(script_path: str) -> int:
    """Walk AST and accumulate parameter counts from layer constructors."""
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=script_path)
    except Exception:
        return 0

    # Pre-collect module-level constants (INPUT_DIM = 1024, etc.)
    consts = _collect_module_constants(tree)

    total = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func

        # Resolve the leaf class name (e.g., "Linear" from "nn.Linear")
        class_name: None | str = None
        if isinstance(func, ast.Attribute):
            class_name = func.attr
        elif isinstance(func, ast.Name):
            class_name = func.id

        if class_name is None:
            continue

        try:
            if class_name == "Linear":
                in_f = _get_int_arg(node, 0, "in_features", consts)
                out_f = _get_int_arg(node, 1, "out_features", consts)
                bias = _get_bool_kwarg(node, "bias", True)
                if in_f and out_f:
                    total += in_f * out_f
                    if bias:
                        total += out_f

            elif class_name == "Embedding":
                num = _get_int_arg(node, 0, "num_embeddings", consts)
                dim = _get_int_arg(node, 1, "embedding_dim", consts)
                if num and dim:
                    total += num * dim

            elif class_name in ("Conv2d", "ConvTranspose2d"):
                c_in = _get_int_arg(node, 0, "in_channels", consts)
                c_out = _get_int_arg(node, 1, "out_channels", consts)
                k = _kernel_size(node, consts)
                bias = _get_bool_kwarg(node, "bias", True)
                if c_in and c_out:
                    total += c_in * c_out * k * k
                    if bias:
                        total += c_out

            elif class_name == "Conv1d":
                c_in = _get_int_arg(node, 0, "in_channels", consts)
                c_out = _get_int_arg(node, 1, "out_channels", consts)
                k = _kernel_size(node, consts)
                bias = _get_bool_kwarg(node, "bias", True)
                if c_in and c_out:
                    total += c_in * c_out * k
                    if bias:
                        total += c_out

            elif class_name == "Conv3d":
                c_in = _get_int_arg(node, 0, "in_channels", consts)
                c_out = _get_int_arg(node, 1, "out_channels", consts)
                k = _kernel_size(node, consts)
                bias = _get_bool_kwarg(node, "bias", True)
                if c_in and c_out:
                    total += c_in * c_out * k * k * k
                    if bias:
                        total += c_out

            elif class_name in ("LayerNorm",):
                if node.args:
                    a = node.args[0]
                    dim = _eval_const_expr(a, consts)
                    if (
                        not isinstance(dim, int)
                        and isinstance(a, ast.Tuple)
                        and a.elts
                    ):
                        dim = _eval_const_expr(a.elts[-1], consts)
                    if isinstance(dim, int) and dim:
                        total += 2 * dim  # weight + bias

            elif class_name in (
                "BatchNorm1d",
                "BatchNorm2d",
                "BatchNorm3d",
                "GroupNorm",
            ):
                num = _get_int_arg(
                    node, 0, "num_features", consts
                ) or _get_int_arg(node, 1, "num_channels", consts)
                if num:
                    total += 2 * num  # weight + bias

            elif class_name == "MultiheadAttention":
                embed_dim = _get_int_arg(node, 0, "embed_dim", consts)
                if embed_dim:
                    total += 4 * embed_dim * embed_dim + 4 * embed_dim

            elif class_name == "GRU":
                in_f = _get_int_arg(node, 0, "input_size", consts)
                hid = _get_int_arg(node, 1, "hidden_size", consts)
                if in_f and hid:
                    total += 3 * (in_f * hid + hid * hid + 2 * hid)

            elif class_name == "LSTM":
                in_f = _get_int_arg(node, 0, "input_size", consts)
                hid = _get_int_arg(node, 1, "hidden_size", consts)
                if in_f and hid:
                    total += 4 * (in_f * hid + hid * hid + 2 * hid)

            elif class_name == "RNN":
                in_f = _get_int_arg(node, 0, "input_size", consts)
                hid = _get_int_arg(node, 1, "hidden_size", consts)
                if in_f and hid:
                    total += in_f * hid + hid * hid + 2 * hid

        except Exception:
            continue  # never crash on a single layer

    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_params(script_path: str, findings: CodeFindings) -> ParamEstimate:
    """Estimate total parameter count and bytes from a script.

    Priority:
      1. Registry match (from_pretrained)
      2. Layer counting from AST
      3. Fallback: 0 params, source="unknown"
    """
    bytes_per_param = _resolve_bytes_per_param(findings)

    # Strategy 1 — registry
    registry_est = _estimate_from_registry(findings, bytes_per_param)
    if registry_est is not None:
        return registry_est

    # Strategy 2 — layer counting
    num_params = _count_params_from_layers(script_path)
    if num_params > 0:
        return ParamEstimate(
            num_params=num_params,
            bytes_per_param=bytes_per_param,
            source="layer_count",
            model_description="Custom model (AST layer count)",
        )

    # Strategy 3 — unknown
    return ParamEstimate(
        num_params=0,
        bytes_per_param=bytes_per_param,
        source="unknown",
        model_description="Unknown model",
    )
