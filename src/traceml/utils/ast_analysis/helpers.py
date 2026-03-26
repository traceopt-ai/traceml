"""Shared AST/path/value helpers for static training-script analysis."""

import ast
from pathlib import Path
from typing import Dict, List, Optional

from traceml.utils.ast_analysis.models import ScriptLocation


def build_import_map(tree: ast.AST) -> Dict[str, str]:
    """Return local-name -> fully-qualified import mapping."""
    mapping: Dict[str, str] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname if alias.asname else alias.name
                mapping[local] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local = alias.asname if alias.asname else alias.name
                fqn = f"{module}.{alias.name}" if module else alias.name
                mapping[local] = fqn

    return mapping


def collect_string_constants(tree: ast.AST) -> Dict[str, str]:
    """Collect simple ``NAME = "string"`` assignments across the file."""
    consts: Dict[str, str] = {}

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            consts[node.targets[0].id] = node.value.value

    return consts


def build_parent_map(tree: ast.AST) -> Dict[int, ast.AST]:
    """Return node-id -> parent mapping."""
    parents: Dict[int, ast.AST] = {}

    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    return parents


def fqn(call: ast.Call, imports: Dict[str, str]) -> Optional[str]:
    """Resolve an ``ast.Call`` function to a likely fully-qualified name."""
    func = call.func

    if isinstance(func, ast.Name):
        return imports.get(func.id, func.id)

    if isinstance(func, ast.Attribute):
        chain: List[str] = []
        cur = func
        while isinstance(cur, ast.Attribute):
            chain.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            chain.append(cur.id)
            chain.reverse()
            root_resolved = imports.get(chain[0], chain[0])
            tail = ".".join(chain[1:])
            return f"{root_resolved}.{tail}" if tail else root_resolved

    return None


def kw_const(call: ast.Call, name: str):
    """Return constant value of keyword *name*, or ``None``."""
    for kw in call.keywords:
        if kw.arg == name:
            return (
                kw.value.value if isinstance(kw.value, ast.Constant) else None
            )
    return None


def kw_repr(call: ast.Call, name: str) -> Optional[str]:
    """Return a compact display representation for keyword *name*."""
    for kw in call.keywords:
        if kw.arg == name:
            if isinstance(kw.value, ast.Constant):
                return repr(kw.value.value)
            if isinstance(kw.value, ast.Name):
                return kw.value.id
            return "<dynamic>"
    return None


def has_kw(call: ast.Call, name: str) -> bool:
    """Return whether *call* contains a keyword argument named *name*."""
    return any(kw.arg == name for kw in call.keywords)


def src_line(lines: List[str], lineno: int) -> str:
    """Return the raw source line for *lineno*, or an empty string."""
    return lines[lineno - 1] if 1 <= lineno <= len(lines) else ""


def location(path: str, node: ast.AST, lines: List[str]) -> ScriptLocation:
    """Build a ``ScriptLocation`` for *node*."""
    ln = getattr(node, "lineno", 0)
    col = getattr(node, "col_offset", 0)
    return ScriptLocation(
        file_path=path,
        line=ln,
        col=col,
        text=src_line(lines, ln),
    )


def safe_int(val) -> Optional[int]:
    """Best-effort ``int`` conversion."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def safe_float(val) -> Optional[float]:
    """Best-effort ``float`` conversion."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def enclosing_function_name(
    node: ast.AST,
    parent_map: Dict[int, ast.AST],
) -> Optional[str]:
    """Return the nearest enclosing function or method name."""
    cur = parent_map.get(id(node))
    while cur is not None:
        if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return cur.name
        cur = parent_map.get(id(cur))
    return None


def phase_hint_for_node(
    node: ast.AST,
    parent_map: Dict[int, ast.AST],
) -> Optional[str]:
    """Infer a likely phase hint from the enclosing function name."""
    fn = (enclosing_function_name(node, parent_map) or "").lower()

    if fn in {"training_step", "train_step", "train_dataloader"}:
        return "train"
    if fn in {"validation_step", "val_step", "val_dataloader", "evaluate"}:
        return "validation"
    if fn in {"test_step", "test_dataloader", "predict_step"}:
        return "test"
    if "train" in fn or fn == "fit":
        return "train"
    if "valid" in fn or "eval" in fn:
        return "validation"
    if "test" in fn or "predict" in fn:
        return "test"
    return None


def extract_assigned_name(
    node: ast.AST,
    parent_map: Dict[int, ast.AST],
) -> Optional[str]:
    """Return variable name if *node* is the RHS of ``name = <node>``."""
    parent = parent_map.get(id(node))
    if not isinstance(parent, ast.Assign):
        return None
    if len(parent.targets) != 1:
        return None
    target = parent.targets[0]
    if isinstance(target, ast.Name):
        return target.id
    return None


def is_in_training_loop(
    node: ast.AST,
    parent_map: Dict[int, ast.AST],
) -> bool:
    """Heuristic check for whether *node* sits inside a training-like loop."""
    phase = phase_hint_for_node(node, parent_map)
    if phase == "train":
        return True

    cur = parent_map.get(id(node))
    while cur is not None:
        if isinstance(cur, (ast.For, ast.While)):
            for child in ast.walk(cur):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Attribute):
                        attr = child.func.attr
                        if attr in ("backward", "zero_grad", "step"):
                            return True
                    elif isinstance(child.func, ast.Name):
                        fn_name = child.func.id.lower()
                        if "backward" in fn_name or "optim" in fn_name:
                            return True
        cur = parent_map.get(id(cur))
    return False


def extract_autocast_dtype(node: ast.Call) -> Optional[str]:
    """Pull the dtype name from an ``autocast(dtype=...)`` call."""
    for kw in node.keywords:
        if kw.arg == "dtype":
            return dtype_attr_to_str(kw.value)
    if len(node.args) >= 2:
        return dtype_attr_to_str(node.args[1])
    return None


def dtype_attr_to_str(node: ast.AST) -> Optional[str]:
    """Convert common dtype AST nodes into a display string."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def is_cudnn_attr(node: ast.Attribute) -> bool:
    """Return True if *node* represents ``torch.backends.cudnn.benchmark``."""
    parts: List[str] = [node.attr]
    cur = node.value
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    parts.reverse()
    joined = ".".join(parts)
    return "cudnn" in joined and "benchmark" in joined


def read_python_source(path: Path) -> str:
    """Read Python source from *path* using UTF-8."""
    return path.read_text(encoding="utf-8")
