"""Compact, payload-only evidence text for the end-of-run terminal card.

The formatters in this module do not diagnose a run or calculate any new
values.  They select short labels for structured diagnosis evidence already
present in ``final_summary.json``.  If a compatible or older payload lacks
the fields a compact form needs, the original stored diagnosis summary remains
the fallback.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from traceml_ai.reporting.terminal_card.common import format_scope

_NORMAL_KINDS = frozenset({"NORMAL", "BALANCED"})


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping, or an empty mapping for malformed payload blocks."""
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    """Return a non-string sequence, or an empty tuple."""
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    return ()


def _float(value: Any) -> Optional[float]:
    """Return a numeric payload value as a float when possible."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> Optional[int]:
    """Return a numeric payload value as an integer when possible."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _group_rows(section: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(_mapping(section.get("groups")).get("rows"))


def _identity_for_rank(
    section: Mapping[str, Any], rank: Optional[int]
) -> Mapping[str, Any]:
    """Resolve a stored rank through existing grouped-row identities."""
    if rank is None:
        return {}
    direct = _mapping(
        _mapping(_group_rows(section).get(str(rank))).get("identity")
    )
    if _int(direct.get("global_rank")) == rank:
        return direct
    for row in _group_rows(section).values():
        identity = _mapping(_mapping(row).get("identity"))
        if _int(identity.get("global_rank")) == rank:
            return identity
    return direct


def _system_scope(diagnosis: Mapping[str, Any]) -> Optional[str]:
    """Return the existing structured System node/GPU scope, if any."""
    scope = _mapping(_mapping(diagnosis.get("evidence")).get("scope"))
    node = scope.get("node_rank", scope.get("node"))
    return format_scope(
        node=_int(node),
        gpu=(
            _int(scope.get("gpu_idx")) if scope.get("level") == "gpu" else None
        ),
    )


def _rank_scope(
    section: Mapping[str, Any], rank: Optional[int]
) -> Optional[str]:
    if rank is None:
        return None
    identity = _identity_for_rank(section, rank)
    return format_scope(rank=rank, node=_int(identity.get("node_rank")))


def _diagnosis_rank(
    diagnosis: Mapping[str, Any], evidence: Mapping[str, Any]
) -> Optional[int]:
    """Choose a rank explicitly named by the stored diagnosis evidence."""
    for key in (
        "rank_gpu_memory_pressure_rank",
        "highest_overhang_rank",
        "highest_rss_rank",
        "rank",
    ):
        rank = _int(evidence.get(key))
        if rank is not None:
            return rank
    for value in _sequence(diagnosis.get("ranks")):
        rank = _int(value)
        if rank is not None:
            return rank
    return None


def _pct(value: Any) -> Optional[str]:
    numeric = _float(value)
    if numeric is None:
        return None
    return f"{numeric:.1f}%"


def _bytes(value: Any) -> Optional[str]:
    """Format an existing byte value compactly without deriving a metric."""
    numeric = _float(value)
    if numeric is None:
        return None
    sign = "+" if numeric > 0.0 else ""
    magnitude = abs(numeric)
    if magnitude < 100_000_000.0:
        return f"{sign}{magnitude / 1e6:.1f} MB"
    return f"{sign}{magnitude / 1e9:.1f} GB"


def _scope_text(text: Optional[str], scope: Optional[str]) -> Optional[str]:
    if not text:
        return None
    return f"{text} · {scope}" if scope else text


def _system_evidence(
    diagnosis: Mapping[str, Any], evidence: Mapping[str, Any]
) -> Optional[str]:
    kind = str(diagnosis.get("kind") or "")
    fields = {
        "VERY_HIGH_GPU_MEMORY": ("gpu_mem_peak_percent", "GPU memory peak"),
        "HIGH_GPU_MEMORY": ("gpu_mem_peak_percent", "GPU memory peak"),
        "HIGH_GPU_TEMPERATURE": ("gpu_temp_peak_c", "GPU temperature peak"),
        "HIGH_GPU_POWER": ("gpu_power_avg_limit_percent", "GPU power"),
        "HIGH_HOST_MEMORY": ("ram_peak_percent", "Host RAM peak"),
        "HIGH_CPU": ("cpu_avg_percent", "CPU"),
        "LOW_GPU_UTILIZATION": ("gpu_util_avg_percent", "GPU utilization"),
        "MODERATE_GPU_UTILIZATION": (
            "gpu_util_avg_percent",
            "GPU utilization",
        ),
    }
    entry = fields.get(kind)
    if entry is None:
        return None
    value = _float(evidence.get(entry[0]))
    if value is None:
        return None
    if kind == "HIGH_GPU_TEMPERATURE":
        text = f"{entry[1]} {value:.1f}C"
    elif kind == "HIGH_GPU_POWER":
        text = f"{entry[1]} {value:.1f}% of limit"
    else:
        text = f"{entry[1]} {value:.1f}%"
    return _scope_text(text, _system_scope(diagnosis))


def _process_evidence(
    section: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> Optional[str]:
    kind = str(diagnosis.get("kind") or "")
    metric = str(diagnosis.get("metric") or "")
    scope = _rank_scope(section, _diagnosis_rank(diagnosis, evidence))
    if kind in {"HIGH_PROCESS_GPU_MEMORY", "VERY_HIGH_PROCESS_GPU_MEMORY"}:
        reserved = "reserved" in metric
        peak_metric = (
            "gpu_mem_reserved_peak_percent"
            if reserved
            else "gpu_mem_used_peak_percent"
        )
        value = _pct(evidence.get(peak_metric))
        return _scope_text(
            (
                f"CUDA {'reserved' if reserved else 'used'} peak {value}"
                if value
                else None
            ),
            scope,
        )
    if kind == "GPU_MEMORY_RESERVED_OVERHANG":
        ratio = _float(evidence.get("gpu_mem_reserved_overhang_ratio"))
        return _scope_text(
            (
                f"CUDA reserved/allocated {ratio:.2f}x"
                if ratio is not None
                else None
            ),
            scope,
        )
    if kind == "RANK_GPU_MEMORY_IMBALANCE":
        value = _pct(evidence.get(metric))
        family = "reserved" if "reserved" in metric else "used"
        return _scope_text(
            f"CUDA {family} imbalance {value}" if value else None,
            scope,
        )
    if kind == "HIGH_PROCESS_RSS":
        value = _pct(evidence.get("ram_peak_percent"))
        return _scope_text(f"RSS peak {value}" if value else None, scope)
    if kind == "HIGH_PROCESS_CPU":
        value = _pct(evidence.get("cpu_capacity_percent"))
        return _scope_text(f"CPU capacity {value}" if value else None, scope)
    return None


def _step_memory_evidence(
    section: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> Optional[str]:
    kind = str(diagnosis.get("kind") or "")
    if kind == "NO_GPU":
        # The heading already states NO GPU; keep its card evidence compact.
        return "Step memory uses torch-based GPU memory telemetry."
    metric = str(diagnosis.get("metric") or "")
    family = "CUDA reserved" if "reserved" in metric else "CUDA allocated"
    scope = _rank_scope(section, _diagnosis_rank(diagnosis, evidence))
    if kind == "HIGH_PRESSURE":
        pressure_frac = _float(evidence.get("pressure_frac"))
        value = (
            _pct(pressure_frac * 100.0) if pressure_frac is not None else None
        )
        return _scope_text(
            f"{family} pressure {value}" if value else None, scope
        )
    if kind == "IMBALANCE":
        skew_pct = _float(evidence.get("skew_pct"))
        pressure_frac = _float(evidence.get("pressure_frac"))
        skew = _pct(skew_pct * 100.0) if skew_pct is not None else None
        pressure = (
            _pct(pressure_frac * 100.0) if pressure_frac is not None else None
        )
        if skew is None or pressure is None:
            return None
        return _scope_text(
            f"{family} skew {skew} · pressure {pressure}",
            scope,
        )
    if kind in {"CREEP_CONFIRMED", "CREEP_EARLY"}:
        growth = _bytes(evidence.get("overall_abs_delta_bytes"))
        growth_fraction = _float(evidence.get("overall_worst_growth_pct"))
        growth_pct = (
            _pct(growth_fraction * 100.0)
            if growth_fraction is not None
            else None
        )
        if growth is not None or growth_pct is not None:
            values = " ".join(
                value
                for value in (
                    growth,
                    f"({growth_pct})" if growth_pct else None,
                )
                if value
            )
            label = (
                "Memory creep"
                if kind == "CREEP_CONFIRMED"
                else "Memory rising"
            )
            return _scope_text(f"{label} {values}", scope)
        note = str(evidence.get("note") or "").strip()
        return _scope_text(note or None, scope)
    return None


def format_run_evidence(
    section_name: str,
    section: Mapping[str, Any],
) -> Optional[str]:
    """Return compact Run-card evidence from an existing section payload.

    ``section_name`` selects only presentation terminology.  No diagnosis or
    aggregate is recalculated.  Unknown, incomplete, and older payloads keep
    their stored diagnosis summary, which is the stable compatibility path.
    """
    diagnosis = _mapping(section.get("diagnosis"))
    if str(diagnosis.get("kind") or "NO_DATA") in _NORMAL_KINDS:
        return None
    evidence = _mapping(diagnosis.get("evidence"))
    if section_name == "system":
        compact = _system_evidence(diagnosis, evidence)
    elif section_name == "process":
        compact = _process_evidence(section, diagnosis, evidence)
    elif section_name == "step_memory":
        compact = _step_memory_evidence(section, diagnosis, evidence)
    else:
        compact = None
    if compact:
        return compact
    note = str(evidence.get("note") or "").strip()
    summary = str(diagnosis.get("summary") or "").strip()
    if note and note not in summary:
        summary = f"{summary} {note}".strip()
    if not summary:
        labels = {
            "system": "System",
            "process": "Process",
            "step_memory": "Step Memory",
        }
        summary = (
            f"{labels.get(section_name, 'Section')} telemetry unavailable."
        )
    scope = (
        _system_scope(diagnosis)
        if section_name == "system"
        else _rank_scope(section, _diagnosis_rank(diagnosis, evidence))
    )
    return _scope_text(summary, scope)


__all__ = ["format_run_evidence"]
