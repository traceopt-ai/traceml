"""Static parameter estimator — walks AST to count model parameters.

Three strategies applied in order of confidence
------------------------------------------------

1. **Registry lookup** (highest confidence)
   If the script calls ``from_pretrained("bert-base-uncased")`` and the model
   name is in ``model_registry``, the exact known parameter count is returned.
   No AST layer walking is needed.

2. **Torchvision factory lookup** (high confidence)
   If the script calls a recognised ``torchvision.models`` factory function
   such as ``vit_b_16()`` or ``resnet50()``, the parameter count is looked up
   from a separate factory table in the registry.

3. **Module-aware layer counting** (medium-high confidence)
   For custom ``nn.Module`` subclasses, only the ``__init__`` method body is
   walked. This avoids the classic double-count bug where the same layer
   appears in both ``__init__`` and ``forward()``.

   Improvements over a naive full-tree walk:
   * **No double-counting** — ``self.fc(x)`` in ``forward()`` is never
     mistaken for a new layer constructor.
   * **Self-attribute constants** — ``self.hidden = 512`` assigned inside
     ``__init__`` is resolved when it later appears as a dimension argument
     (e.g. ``nn.Linear(784, self.hidden)``).
   * **Recursive submodules** — ``self.block = Block()`` recurses into
     ``Block.__init__`` with a visited-set guard against infinite loops.
   * **Container unpacking** — ``nn.Sequential(...)``, ``nn.ModuleList([...])``
     and ``nn.ModuleDict({...})`` are enumerated and their contents counted.
   * **Multi-layer RNNs** — ``num_layers=N`` and ``bidirectional`` are
     handled correctly for LSTM, GRU and plain RNN.
   * **Framework subclasses** — ``pl.LightningModule``, ``L.LightningModule``
     and similar wrappers are treated as ``nn.Module`` subclasses so nested
     models inside them are discovered.

4. **Flat AST walk** (fallback / medium confidence)
   If no ``nn.Module`` subclasses are found (e.g. a purely functional script),
   the original approach of walking every Call node in the file is used.

Never crashes — returns a valid ``ParamEstimate(num_params=0)`` on any failure.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Set

from traceml.utils.ast_analysis.model_registry import (
    ModelSpec,
    lookup_model,
    lookup_torchvision_factory,
)
from traceml.utils.ast_analysis.scanner import CodeFindings

# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class ParamEstimate:
    """Result of a static parameter count estimation.

    Attributes
    ----------
    num_params:
        Estimated total number of trainable parameters.
    bytes_per_param:
        Storage size per parameter in bytes (4 = fp32, 2 = fp16/bf16).
    source:
        How the estimate was produced:
        ``"registry"``    — exact count from the known-model registry.
        ``"layer_count"`` — counted from AST layer constructors.
        ``"unknown"``     — could not determine (num_params will be 0).
    model_description:
        Human-readable model name shown in the CLI output.
    """

    num_params: int
    bytes_per_param: int  # 4 = fp32, 2 = fp16/bf16
    source: str  # "registry" | "layer_count" | "unknown"
    model_description: str = ""

    @property
    def total_bytes(self) -> int:
        """Total weight memory in bytes (num_params × bytes_per_param)."""
        return self.num_params * self.bytes_per_param

    @property
    def total_gb(self) -> float:
        """Total weight memory in gibibytes (GiB)."""
        return self.total_bytes / (1024**3)


# ---------------------------------------------------------------------------
# 1. Precision detection
# ---------------------------------------------------------------------------


def _resolve_bytes_per_param(findings: CodeFindings) -> int:
    """Infer bytes-per-parameter from precision findings in the script.

    Returns 2 when fp16 or bfloat16 usage is detected (autocast, .half(),
    ``torch_dtype=torch.float16``, etc.), 4 otherwise (fp32 default).
    """
    for p in findings.precision:
        if p.dtype_str in ("float16", "bfloat16", "half"):
            return 2
        if p.kind in ("half", "bfloat16"):
            return 2
    return 4


# ---------------------------------------------------------------------------
# 2. Registry-based estimation (HuggingFace from_pretrained)
# ---------------------------------------------------------------------------


def _estimate_from_registry(
    findings: CodeFindings, bytes_per_param: int
) -> Optional[ParamEstimate]:
    """Try to match a ``from_pretrained()`` call to the known model registry.

    Walks all ``ModelFinding`` entries produced by the AST scanner and
    returns the first one whose ``model_name`` resolves in the registry.
    Returns ``None`` when no registry match is found.
    """
    for mf in findings.models:
        if mf.kind == "from_pretrained" and mf.model_name:
            spec: Optional[ModelSpec] = lookup_model(mf.model_name)
            if spec:
                # Honour lower precision if the script uses fp16/bf16
                bpp = min(bytes_per_param, spec.bytes_per_param)
                return ParamEstimate(
                    num_params=spec.num_params,
                    bytes_per_param=bpp,
                    source="registry",
                    model_description=spec.description or mf.model_name,
                )
    return None


# ---------------------------------------------------------------------------
# 3. Torchvision factory estimation
# ---------------------------------------------------------------------------


def _estimate_from_torchvision(
    tree: ast.AST, bytes_per_param: int
) -> Optional[ParamEstimate]:
    """Detect torchvision model factory calls and look them up.

    Handles patterns such as::

        model = vit_b_16(num_classes=365)
        model = torchvision.models.resnet50(pretrained=True)

    The leaf function name is resolved and looked up in the torchvision
    factory table.  Returns ``None`` when no factory call is recognised.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Resolve the leaf name of the called function
        leaf: Optional[str] = None
        if isinstance(func, ast.Attribute):
            leaf = func.attr
        elif isinstance(func, ast.Name):
            leaf = func.id
        if leaf is None:
            continue
        spec = lookup_torchvision_factory(leaf)
        if spec:
            return ParamEstimate(
                num_params=spec.num_params,
                bytes_per_param=bytes_per_param,  # script precision takes precedence
                source="layer_count",
                model_description=spec.description,
            )
    return None


# ---------------------------------------------------------------------------
# Constant evaluation helpers
# ---------------------------------------------------------------------------


def _eval_const_expr(
    node: ast.AST,
    consts: Dict[str, int],
    self_consts: Optional[Dict[str, int]] = None,
) -> Optional[int]:
    """Recursively evaluate a compile-time integer constant expression.

    Supported node types
    --------------------
    * ``ast.Constant``    — integer literal (``512``, ``3``, …)
    * ``ast.Name``        — bare name lookup in *consts*
                           (handles module-level ``HIDDEN = 256``)
    * ``ast.Attribute``   — ``self.ATTR`` lookup in *self_consts*
                           (handles ``self.hidden = 512`` pattern)
    * ``ast.BinOp``       — ``+``, ``-``, ``*``, ``//``, ``/``
                           (handles ``HIDDEN = BASE * 4``)

    Returns ``None`` for anything more complex (variables, function calls, …).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value

    if isinstance(node, ast.Name):
        return consts.get(node.id)

    # self.attr — look up in the per-class self_consts dict
    if (
        self_consts is not None
        and isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return self_consts.get(node.attr)

    if isinstance(node, ast.BinOp):
        left = _eval_const_expr(node.left, consts, self_consts)
        right = _eval_const_expr(node.right, consts, self_consts)
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
                return int(left // right)
        except Exception:
            return None

    return None


def _int_arg(
    call: ast.Call,
    pos: int,
    kw: str,
    consts: Dict[str, int],
    self_consts: Optional[Dict[str, int]] = None,
) -> Optional[int]:
    """Return the resolved integer at positional slot *pos* or keyword *kw*.

    Tries *pos* first (positional argument), then falls back to searching
    the keyword arguments for *kw*.  Both paths go through
    :func:`_eval_const_expr` so named constants and ``self.X`` refs work.
    """
    if pos < len(call.args):
        v = _eval_const_expr(call.args[pos], consts, self_consts)
        if isinstance(v, int):
            return v
    for k in call.keywords:
        if k.arg == kw:
            v = _eval_const_expr(k.value, consts, self_consts)
            if isinstance(v, int):
                return v
    return None


def _bool_kwarg(call: ast.Call, name: str, default: bool = True) -> bool:
    """Return the boolean value of a keyword argument, or *default*."""
    for kw in call.keywords:
        if kw.arg == name and isinstance(kw.value, ast.Constant):
            return bool(kw.value.value)
    return default


def _kernel_size(
    call: ast.Call,
    consts: Dict[str, int],
    self_consts: Optional[Dict[str, int]] = None,
    pos: int = 2,
) -> int:
    """Extract ``kernel_size`` from a Conv layer constructor.

    Handles:
    * Positional integer:          ``nn.Conv2d(3, 64, 3)``
    * Named constant:              ``nn.Conv2d(3, 64, K)``
    * Tuple (uses first element):  ``nn.Conv2d(3, 64, (3, 3))``
    * Keyword arg:                 ``nn.Conv2d(3, 64, kernel_size=3)``

    Falls back to 3 when unresolvable (safe/common default).
    """
    if pos < len(call.args):
        a = call.args[pos]
        v = _eval_const_expr(a, consts, self_consts)
        if isinstance(v, int):
            return v
        if isinstance(a, ast.Tuple) and a.elts:
            v = _eval_const_expr(a.elts[0], consts, self_consts)
            if isinstance(v, int):
                return v
    for kw in call.keywords:
        if kw.arg == "kernel_size":
            v = _eval_const_expr(kw.value, consts, self_consts)
            if isinstance(v, int):
                return v
            if isinstance(kw.value, ast.Tuple) and kw.value.elts:
                v = _eval_const_expr(kw.value.elts[0], consts, self_consts)
                if isinstance(v, int):
                    return v
    return 3  # conservative fallback


# ---------------------------------------------------------------------------
# Module-level and self-attribute constant collection
# ---------------------------------------------------------------------------


def _collect_module_constants(tree: ast.AST) -> Dict[str, int]:
    """Build a map of integer-valued ``NAME = <expr>`` assignments.

    Scans *all* scopes (module level, function bodies, class methods) so
    that patterns like ``model_name = "bert-base"`` inside ``main()`` are
    also captured.  Runs two passes so forward references resolve
    (e.g. ``WIDE = BASE * 4`` where ``BASE`` is defined earlier).

    Returns a ``{name: int_value}`` dict.
    """
    consts: Dict[str, int] = {}
    for _ in range(2):  # second pass resolves cross-references
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id not in consts  # first assignment wins
            ):
                v = _eval_const_expr(node.value, consts)
                if isinstance(v, int):
                    consts[node.targets[0].id] = v
    return consts


def _collect_self_constants(
    init_fn: ast.FunctionDef,
    consts: Dict[str, int],
) -> Dict[str, int]:
    """Collect ``self.X = <integer-expr>`` assignments inside ``__init__``.

    Allows dimension arguments like ``nn.Linear(784, self.hidden)`` to
    resolve when ``self.hidden = 512`` was set earlier in the same method.

    The returned dict is passed as *self_consts* whenever layer dims are
    evaluated for this class.
    """
    sc: Dict[str, int] = {}
    for node in ast.walk(init_fn):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "self"
        ):
            # Evaluate RHS with current module consts + already-seen self consts
            v = _eval_const_expr(node.value, consts, sc)
            if isinstance(v, int):
                sc[node.targets[0].attr] = v
    return sc


# ---------------------------------------------------------------------------
# nn.Module class discovery
# ---------------------------------------------------------------------------

# Base class leaf names that indicate an nn.Module-compatible class.
# Includes PyTorch Lightning wrappers and similar framework bases so that
# models defined as subclasses of these are also counted.
_MODULE_BASE_NAMES: FrozenSet[str] = frozenset(
    {
        "Module",  # nn.Module
        "LightningModule",  # pytorch_lightning / lightning
        "TrainingMixin",  # some custom base patterns
    }
)


def _leaf_name(node: ast.expr) -> Optional[str]:
    """Return the bare identifier of a ``Name`` or ``Attribute`` node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _base_names(classdef: ast.ClassDef) -> List[str]:
    """Collect all base class leaf names for *classdef*."""
    return [n for n in (_leaf_name(b) for b in classdef.bases) if n]


def _is_nn_module(
    classdef: ast.ClassDef,
    all_classes: Dict[str, ast.ClassDef],
    _visiting: Optional[FrozenSet[str]] = None,
) -> bool:
    """Return True when *classdef* ultimately inherits from ``nn.Module``.

    Recursively checks base classes defined in the same file (handles
    intermediate base classes like a custom ``BaseModel(nn.Module)``).
    Also recognises Lightning and other popular framework wrappers via
    ``_MODULE_BASE_NAMES``.

    The *_visiting* frozenset prevents infinite recursion for pathological
    class hierarchies.
    """
    if _visiting is None:
        _visiting = frozenset()
    if classdef.name in _visiting:
        return False  # recursion guard
    _visiting = _visiting | {classdef.name}

    for bname in _base_names(classdef):
        # Direct match against known framework base class names
        if bname in _MODULE_BASE_NAMES:
            return True
        # Recurse into base classes defined in this file
        if bname in all_classes:
            if _is_nn_module(all_classes[bname], all_classes, _visiting):
                return True
    return False


def _get_init(classdef: ast.ClassDef) -> Optional[ast.FunctionDef]:
    """Return the ``__init__`` method of *classdef*, or ``None``."""
    for stmt in classdef.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
            return stmt
    return None


# ---------------------------------------------------------------------------
# Layer parameter counting
# ---------------------------------------------------------------------------

# Container class names whose contents should be recursively unpacked.
_CONTAINER_NAMES: FrozenSet[str] = frozenset(
    {"Sequential", "ModuleList", "ModuleDict"}
)


def _count_single_layer(
    call: ast.Call,
    layer_name: str,
    consts: Dict[str, int],
    self_consts: Optional[Dict[str, int]],
) -> int:
    """Return the parameter count for a single recognised PyTorch layer call.

    Supported layers and their parameter formulas
    ----------------------------------------------
    ``Linear(in, out)``
        ``in×out + out`` (+ bias when ``bias=True``, which is the default)

    ``Embedding(num, dim)``
        ``num×dim``

    ``Conv2d / ConvTranspose2d(c_in, c_out, k)``
        ``c_in×c_out×k² + c_out``

    ``Conv1d(c_in, c_out, k)``
        ``c_in×c_out×k + c_out``

    ``Conv3d(c_in, c_out, k)``
        ``c_in×c_out×k³ + c_out``

    ``LayerNorm(shape)``
        ``2×shape`` (weight + bias)

    ``BatchNorm1d/2d/3d(num_features)``
        ``2×num_features``

    ``GroupNorm(groups, num_channels)``
        ``2×num_channels``

    ``MultiheadAttention(embed_dim, heads)``
        ``4×embed_dim² + 4×embed_dim``
        (Q/K/V/out projections each of size ``embed_dim×embed_dim``)

    ``LSTM(input_size, hidden, num_layers, bidirectional)``
        Layer 0:   ``4×(input×hidden + hidden²+ 2×hidden) × dirs``
        Layers 1‥N: ``4×(dirs×hidden×hidden + hidden²+ 2×hidden) × dirs``

    ``GRU`` / ``RNN``
        Same pattern with 3 (GRU) or 1 (RNN) gate blocks instead of 4.

    Returns 0 for unrecognised layer names (never raises).
    """
    sc = self_consts or {}
    try:
        if layer_name == "Linear":
            in_f = _int_arg(call, 0, "in_features", consts, sc)
            out_f = _int_arg(call, 1, "out_features", consts, sc)
            bias = _bool_kwarg(call, "bias", True)
            if in_f and out_f:
                return in_f * out_f + (out_f if bias else 0)

        elif layer_name == "Embedding":
            num = _int_arg(call, 0, "num_embeddings", consts, sc)
            dim = _int_arg(call, 1, "embedding_dim", consts, sc)
            if num and dim:
                return num * dim

        elif layer_name in ("Conv2d", "ConvTranspose2d"):
            c_in = _int_arg(call, 0, "in_channels", consts, sc)
            c_out = _int_arg(call, 1, "out_channels", consts, sc)
            k = _kernel_size(call, consts, sc)
            bias = _bool_kwarg(call, "bias", True)
            if c_in and c_out:
                return c_in * c_out * k * k + (c_out if bias else 0)

        elif layer_name == "Conv1d":
            c_in = _int_arg(call, 0, "in_channels", consts, sc)
            c_out = _int_arg(call, 1, "out_channels", consts, sc)
            k = _kernel_size(call, consts, sc)
            bias = _bool_kwarg(call, "bias", True)
            if c_in and c_out:
                return c_in * c_out * k + (c_out if bias else 0)

        elif layer_name == "Conv3d":
            c_in = _int_arg(call, 0, "in_channels", consts, sc)
            c_out = _int_arg(call, 1, "out_channels", consts, sc)
            k = _kernel_size(call, consts, sc)
            bias = _bool_kwarg(call, "bias", True)
            if c_in and c_out:
                return c_in * c_out * k * k * k + (c_out if bias else 0)

        elif layer_name == "LayerNorm":
            # LayerNorm(normalized_shape) — shape may be int or tuple
            if call.args:
                a = call.args[0]
                dim = _eval_const_expr(a, consts, sc)
                if (
                    not isinstance(dim, int)
                    and isinstance(a, ast.Tuple)
                    and a.elts
                ):
                    # Use last dimension (most common use-case: LayerNorm([T, D]) → D)
                    dim = _eval_const_expr(a.elts[-1], consts, sc)
                if isinstance(dim, int) and dim:
                    return 2 * dim  # weight + bias

        elif layer_name in ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d"):
            num = _int_arg(call, 0, "num_features", consts, sc)
            if num:
                return 2 * num  # weight + bias

        elif layer_name == "GroupNorm":
            # GroupNorm(num_groups, num_channels)
            num = _int_arg(call, 1, "num_channels", consts, sc)
            if num:
                return 2 * num

        elif layer_name == "MultiheadAttention":
            # 4 projections (Q, K, V, out), each embed_dim × embed_dim + bias
            embed_dim = _int_arg(call, 0, "embed_dim", consts, sc)
            if embed_dim:
                return 4 * embed_dim * embed_dim + 4 * embed_dim

        elif layer_name == "LSTM":
            in_f = _int_arg(call, 0, "input_size", consts, sc)
            hid = _int_arg(call, 1, "hidden_size", consts, sc)
            num_layers = _int_arg(call, 2, "num_layers", consts, sc) or 1
            bidir = _bool_kwarg(call, "bidirectional", False)
            dirs = 2 if bidir else 1
            if in_f and hid:
                # Layer 0: input_size feeds into gates
                p0 = 4 * (in_f * hid + hid * hid + 2 * hid) * dirs
                # Subsequent layers: previous layer's output (dirs×hid) feeds in
                hid_in = dirs * hid
                px = 4 * (hid_in * hid + hid * hid + 2 * hid) * dirs
                return p0 + (num_layers - 1) * px

        elif layer_name == "GRU":
            in_f = _int_arg(call, 0, "input_size", consts, sc)
            hid = _int_arg(call, 1, "hidden_size", consts, sc)
            num_layers = _int_arg(call, 2, "num_layers", consts, sc) or 1
            bidir = _bool_kwarg(call, "bidirectional", False)
            dirs = 2 if bidir else 1
            if in_f and hid:
                p0 = 3 * (in_f * hid + hid * hid + 2 * hid) * dirs
                hid_in = dirs * hid
                px = 3 * (hid_in * hid + hid * hid + 2 * hid) * dirs
                return p0 + (num_layers - 1) * px

        elif layer_name == "RNN":
            in_f = _int_arg(call, 0, "input_size", consts, sc)
            hid = _int_arg(call, 1, "hidden_size", consts, sc)
            num_layers = _int_arg(call, 2, "num_layers", consts, sc) or 1
            bidir = _bool_kwarg(call, "bidirectional", False)
            dirs = 2 if bidir else 1
            if in_f and hid:
                p0 = (in_f * hid + hid * hid + 2 * hid) * dirs
                hid_in = dirs * hid
                px = (hid_in * hid + hid * hid + 2 * hid) * dirs
                return p0 + (num_layers - 1) * px

    except Exception:
        pass  # robustness: never crash on a single unrecognised layer
    return 0


def _count_node_params(
    call: ast.Call,
    consts: Dict[str, int],
    self_consts: Optional[Dict[str, int]],
    all_classes: Dict[str, ast.ClassDef],
    visited: FrozenSet[str],
) -> int:
    """Dispatch a single Call node to the appropriate counter.

    Resolution order:
    1. Container (``Sequential``, ``ModuleList``, ``ModuleDict``) → unpack.
    2. Custom ``nn.Module`` subclass defined in the same file → recurse.
    3. Known PyTorch primitive layer → count directly.
    """
    func = call.func
    if isinstance(func, ast.Attribute):
        name = func.attr
    elif isinstance(func, ast.Name):
        name = func.id
    else:
        return 0  # unresolvable call expression

    # 1. Container — unpack and count contents
    if name in _CONTAINER_NAMES:
        return _count_container_contents(
            call, consts, self_consts, all_classes, visited
        )

    # 2. Custom submodule — recurse into its __init__
    if name in all_classes and name not in visited:
        cd = all_classes[name]
        if _is_nn_module(cd, all_classes):
            return _count_class_params(name, all_classes, consts, visited)

    # 3. PyTorch primitive layer
    return _count_single_layer(call, name, consts, self_consts)


def _count_container_contents(
    call: ast.Call,
    consts: Dict[str, int],
    self_consts: Optional[Dict[str, int]],
    all_classes: Dict[str, ast.ClassDef],
    visited: FrozenSet[str],
) -> int:
    """Count parameters inside a container layer.

    Handles three argument forms:
    * ``nn.Sequential(nn.Linear(...), nn.ReLU(), ...)`` — positional Call args
    * ``nn.ModuleList([nn.Linear(...), ...])``     — list literal element
    * ``nn.ModuleDict({"fc": nn.Linear(...), ...})`` — dict literal values
    """
    total = 0
    for arg in call.args:
        if isinstance(arg, ast.Call):
            # Directly passed Call node: nn.Sequential(nn.Linear(...), ...)
            total += _count_node_params(
                arg, consts, self_consts, all_classes, visited
            )
        elif isinstance(arg, (ast.List, ast.Tuple)):
            # List / tuple literal: nn.ModuleList([...])
            for elt in arg.elts:
                if isinstance(elt, ast.Call):
                    total += _count_node_params(
                        elt, consts, self_consts, all_classes, visited
                    )
        elif isinstance(arg, ast.Dict):
            # Dict literal: nn.ModuleDict({"a": nn.Linear(...), ...})
            for v in arg.values:
                if isinstance(v, ast.Call):
                    total += _count_node_params(
                        v, consts, self_consts, all_classes, visited
                    )
    return total


def _count_class_params(
    class_name: str,
    all_classes: Dict[str, ast.ClassDef],
    consts: Dict[str, int],
    visited: FrozenSet[str],
) -> int:
    """Count all parameters contributed by one ``nn.Module`` subclass.

    Algorithm
    ---------
    1. Locate the class's ``__init__`` method (returns 0 if absent).
    2. Collect ``self.X = int`` assignments to build *self_consts*.
    3. Walk ``__init__``'s body for ``self.<attr> = <Call>`` assignments.
    4. For each unique attribute name (deduplicated to avoid if/else
       double-counting), dispatch to :func:`_count_node_params`.

    The *visited* frozenset prevents infinite recursion when classes
    reference each other (e.g. ``A`` uses ``B`` which uses ``A``).
    """
    visited = visited | {class_name}  # immutable union — no shared state

    classdef = all_classes.get(class_name)
    if classdef is None:
        return 0

    init_fn = _get_init(classdef)
    if init_fn is None:
        return 0

    # Build self-attribute constant map for this class
    self_consts = _collect_self_constants(init_fn, consts)

    seen_attrs: Set[str] = set()
    total = 0

    for node in ast.walk(init_fn):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]

        # We only care about ``self.<attr> = ...`` assignments
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            continue

        attr = target.attr
        # First assignment wins — handles if/else branches without double-counting
        if attr in seen_attrs:
            continue
        seen_attrs.add(attr)

        # RHS must be a Call to be a layer/submodule
        if not isinstance(node.value, ast.Call):
            continue

        try:
            total += _count_node_params(
                node.value, consts, self_consts, all_classes, visited
            )
        except Exception:
            continue  # never crash on a single unresolvable assignment

    return total


# ---------------------------------------------------------------------------
# High-level counting orchestration
# ---------------------------------------------------------------------------


def _count_params_module_aware(tree: ast.AST, consts: Dict[str, int]) -> int:
    """Orchestrate module-aware parameter counting.

    Steps
    -----
    1. Collect all ``ClassDef`` nodes in the file.
    2. Filter to those that inherit from ``nn.Module`` (or framework wrappers).
    3. Identify "root" classes — those *not* used as submodules inside other
       module classes (to avoid counting nested classes twice).
    4. Call :func:`_count_class_params` for each root class.
    """
    all_classes: Dict[str, ast.ClassDef] = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
    }
    if not all_classes:
        return 0

    module_classes = {
        name: cd
        for name, cd in all_classes.items()
        if _is_nn_module(cd, all_classes)
    }
    if not module_classes:
        return 0

    # Determine which classes appear as ``self.X = ChildClass()`` inside
    # another module class — these are submodules and should not be counted
    # independently to prevent double-counting.
    used_as_submodule: Set[str] = set()
    for name in module_classes:
        init_fn = _get_init(all_classes[name])
        if init_fn is None:
            continue
        for node in ast.walk(init_fn):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            child_name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else None
            )
            if (
                child_name
                and child_name in module_classes
                and child_name != name
            ):
                used_as_submodule.add(child_name)

    # Root classes = module classes that are not used inside another module
    roots = [n for n in module_classes if n not in used_as_submodule]
    if not roots:
        roots = list(module_classes)  # safety fallback

    total = 0
    for root in roots:
        total += _count_class_params(root, all_classes, consts, frozenset())
    return total


def _count_params_flat(tree: ast.AST, consts: Dict[str, int]) -> int:
    """Fallback: count parameters by walking every Call node in the file.

    Used when no ``nn.Module`` subclasses are detected (e.g. purely
    functional code using ``torch.nn.functional`` directly, or scripts
    that build models via factory functions not in the registry).

    Each recognised layer name is counted regardless of where it appears
    in the file (module level, function body, class body, etc.).
    """
    total = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id if isinstance(func, ast.Name) else None
        )
        if name is None:
            continue
        try:
            total += _count_single_layer(node, name, consts, None)
        except Exception:
            continue
    return total


def _count_params_from_layers(script_path: str) -> int:
    """Parse *script_path* and return an estimated parameter count.

    Tries module-aware counting first; falls back to flat walking when
    no ``nn.Module`` subclasses are found or the module-aware pass returns 0.
    The torchvision factory check is handled at the ``estimate_params`` level
    since it requires the scanner ``findings`` object.
    """
    try:
        with open(script_path, "r", encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source, filename=script_path)
    except Exception:
        return 0

    consts = _collect_module_constants(tree)

    # Prefer module-aware (no double-count, resolves self-attrs, submodules)
    n = _count_params_module_aware(tree, consts)
    if n > 0:
        return n

    # Fallback for functional-style scripts
    return _count_params_flat(tree, consts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def estimate_params(script_path: str, findings: CodeFindings) -> ParamEstimate:
    """Estimate total parameter count and per-param bytes from *script_path*.

    Priority order
    --------------
    1. HuggingFace registry   — ``from_pretrained("known-model")``
    2. Torchvision factory    — ``vit_b_16()``, ``resnet50()``, …
    3. Module-aware AST count — custom ``nn.Module`` subclasses
    4. Flat AST walk          — functional-style fallback
    5. Unknown                — returns 0 with source ``"unknown"``

    Parameters
    ----------
    script_path:
        Absolute or relative path to the Python training script.
    findings:
        Pre-computed ``CodeFindings`` from the AST scanner (contains
        detected optimizers, precision, model provenance, etc.).
    """
    bytes_per_param = _resolve_bytes_per_param(findings)

    # --- Strategy 1: HuggingFace registry ---
    registry_est = _estimate_from_registry(findings, bytes_per_param)
    if registry_est is not None:
        return registry_est

    # --- Strategy 2: Torchvision factory ---
    try:
        with open(script_path, "r", encoding="utf-8") as fh:
            source = fh.read()
        tree = ast.parse(source, filename=script_path)
        tv_est = _estimate_from_torchvision(tree, bytes_per_param)
        if tv_est is not None:
            return tv_est
    except Exception:
        pass

    # --- Strategies 3 & 4: AST layer counting ---
    num_params = _count_params_from_layers(script_path)
    if num_params > 0:
        return ParamEstimate(
            num_params=num_params,
            bytes_per_param=bytes_per_param,
            source="layer_count",
            model_description="Custom model (AST layer count)",
        )

    # --- Strategy 5: Give up gracefully ---
    return ParamEstimate(
        num_params=0,
        bytes_per_param=bytes_per_param,
        source="unknown",
        model_description="Unknown model",
    )
