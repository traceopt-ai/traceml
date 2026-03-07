import json
import sqlite3
from collections import deque
from typing import Any, Deque, Dict, Optional

import msgspec


def _append_text(path: str, text: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")


def _load_json_or_empty(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return 0.0


def _event_total_ms(by_dev: Any) -> float:
    """
    Sum duration_ms across all devices for one event.
    Malformed entries are ignored.
    """
    if not isinstance(by_dev, dict):
        return 0.0

    total = 0.0
    for stats in by_dev.values():
        if not isinstance(stats, dict):
            continue
        total += _safe_float(stats.get("duration_ms", 0.0))
    return total


def _event_bucket(name: str) -> Optional[str]:
    """
    Map raw event names to canonical step-time buckets.

    Returns one of:
      - dataloader
      - forward
      - backward
      - optimizer
      - step_time
      - None
    """
    n = str(name).lower()

    # Exact / internal aliases first
    if "step_time" in n:
        return "step_time"
    if "dataloader_next" in n:
        return "dataloader"
    if "forward_time" in n:
        return "forward"
    if "backward_time" in n:
        return "backward"
    if "optimizer_step" in n:
        return "optimizer"

    # Permissive fallbacks
    if "data" in n or "dataloader" in n or "input" in n or "batch" in n:
        return "dataloader"
    if "forward" in n or n == "fwd":
        return "forward"
    if "backward" in n or "bwd" in n:
        return "backward"
    if "optim" in n or "optimizer" in n or n in {"step", "update"}:
        return "optimizer"

    return None


def _coerce_rank(db_rank: Any, msg_rank: Any) -> Optional[int]:
    for v in (db_rank, msg_rank):
        try:
            if v is not None:
                return int(v)
        except Exception:
            pass
    return None


def _fmt_ms(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.1f}ms"


def _fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.0f}%"


def _share(num: float, denom: float) -> Optional[float]:
    if denom <= 0.0:
        return None
    return 100.0 * num / denom


def _closest_rank_to_median(rank_to_value: Dict[int, float]) -> Optional[int]:
    if not rank_to_value:
        return None

    vals = sorted(rank_to_value.values())
    n = len(vals)
    if n % 2 == 1:
        median_val = vals[n // 2]
    else:
        median_val = 0.5 * (vals[n // 2 - 1] + vals[n // 2])

    return min(
        rank_to_value.keys(),
        key=lambda r: (
            abs(rank_to_value[r] - median_val),
            rank_to_value[r],
            r,
        ),
    )


def generate_step_time_summary_card(
    db_path: str,
    step_sampler_name: str = "StepTimeSampler",
    max_rows: int = 50_000,
    print_to_stdout: bool = True,
) -> Dict[str, Any]:
    """
    Append a compact end-of-run StepTime summary card beneath the SYSTEM card.

    Summary logic
    -------------
    For each rank, look at its last up to `max_rows` available steps.
    For each step, compute:

        gpu_compute = forward + backward + optimizer
        total_step* = dataloader + max(step_time, gpu_compute)

    Then compare ranks using the average of `total_step*` over that rank's
    analyzed steps.

    - worst rank  = highest avg total_step*
    - median rank = rank closest to median avg total_step*

    Notes
    -----
    - This is intentionally a simple end-of-run summary, not a strict
      per-step aligned cross-rank analysis.
    - `training_steps` is estimated from the maximum observed step id:
          training_steps = max_step_observed + 1
      assuming steps are 0-based.

    Writes/updates:
      - <db_path>_summary_card.txt   (APPEND)
      - <db_path>_summary_card.json  (MERGE under key "step_time")
    """
    conn = sqlite3.connect(db_path)
    dec = msgspec.msgpack.Decoder(type=dict)

    # Per-rank rolling storage of the latest up to `max_rows` unique steps.
    # rank -> {step -> metrics_dict}
    per_rank_steps: Dict[int, Dict[int, Dict[str, float]]] = {}
    # rank -> step insertion order for bounded retention
    per_rank_order: Dict[int, Deque[int]] = {}
    # latest observed step id per rank
    latest_step_by_rank: Dict[int, int] = {}

    cur = conn.execute(
        """
        SELECT rank, payload_mp
        FROM raw_messages
        WHERE sampler = ?
        ORDER BY id ASC;
        """,
        (step_sampler_name,),
    )

    for db_rank, blob in cur:
        if not blob:
            continue

        try:
            msg = dec.decode(blob)
        except Exception:
            continue

        if not isinstance(msg, dict):
            continue

        rank = _coerce_rank(db_rank, msg.get("rank"))
        if rank is None:
            continue

        tables = msg.get("tables")
        if not isinstance(tables, dict):
            continue

        if rank not in per_rank_steps:
            per_rank_steps[rank] = {}
            per_rank_order[rank] = deque()

        step_map = per_rank_steps[rank]
        step_order = per_rank_order[rank]

        for rows in tables.values():
            if not isinstance(rows, list):
                continue

            for row in rows:
                if not isinstance(row, dict):
                    continue

                try:
                    step = int(row.get("step"))
                except Exception:
                    continue

                events = row.get("events")
                if not isinstance(events, dict) or not events:
                    continue

                metrics = {
                    "dataloader": 0.0,
                    "forward": 0.0,
                    "backward": 0.0,
                    "optimizer": 0.0,
                    "step_time": 0.0,
                }

                for evt_name, by_dev in events.items():
                    bucket = _event_bucket(str(evt_name))
                    if bucket is None:
                        continue
                    metrics[bucket] += _event_total_ms(by_dev)

                if (
                    metrics["dataloader"] <= 0.0
                    and metrics["forward"] <= 0.0
                    and metrics["backward"] <= 0.0
                    and metrics["optimizer"] <= 0.0
                    and metrics["step_time"] <= 0.0
                ):
                    continue

                if step not in step_map:
                    step_order.append(step)
                step_map[step] = metrics

                while len(step_order) > max_rows:
                    old_step = step_order.popleft()
                    step_map.pop(old_step, None)

                prev_latest = latest_step_by_rank.get(rank)
                if prev_latest is None or step > prev_latest:
                    latest_step_by_rank[rank] = step

    conn.close()

    def rank_summary(
        step_map: Dict[int, Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        if not step_map:
            return None

        n = len(step_map)
        if n == 0:
            return None

        sum_dl = 0.0
        sum_fwd = 0.0
        sum_bwd = 0.0
        sum_opt = 0.0
        sum_step = 0.0
        sum_total_proxy = 0.0

        for m in step_map.values():
            dl = float(m.get("dataloader", 0.0))
            fwd = float(m.get("forward", 0.0))
            bwd = float(m.get("backward", 0.0))
            opt = float(m.get("optimizer", 0.0))
            step_cpu = float(m.get("step_time", 0.0))
            gpu_compute = fwd + bwd + opt
            total_proxy = dl + max(step_cpu, gpu_compute)

            sum_dl += dl
            sum_fwd += fwd
            sum_bwd += bwd
            sum_opt += opt
            sum_step += step_cpu
            sum_total_proxy += total_proxy

        return {
            "steps_analyzed": float(n),
            "avg_dataloader_ms": sum_dl / n,
            "avg_forward_ms": sum_fwd / n,
            "avg_backward_ms": sum_bwd / n,
            "avg_optimizer_ms": sum_opt / n,
            "avg_step_cpu_ms": sum_step / n,
            "avg_gpu_compute_ms": (sum_fwd + sum_bwd + sum_opt) / n,
            "avg_total_proxy_ms": sum_total_proxy / n,
        }

    per_rank_summary: Dict[int, Dict[str, float]] = {}
    for rank, step_map in per_rank_steps.items():
        s = rank_summary(step_map)
        if s is not None:
            per_rank_summary[rank] = s

    ranks_present = sorted(per_rank_summary.keys())
    latest_step_observed = (
        max(latest_step_by_rank.values()) if latest_step_by_rank else None
    )
    training_steps = (
        int(latest_step_observed) + 1
        if latest_step_observed is not None
        else 0
    )

    worst_rank: Optional[int] = None
    median_rank: Optional[int] = None
    worst_avg_total_proxy_ms: Optional[float] = None
    median_avg_total_proxy_ms: Optional[float] = None
    worst_vs_median_pct: Optional[float] = None

    if per_rank_summary:
        avg_total_by_rank = {
            r: per_rank_summary[r]["avg_total_proxy_ms"]
            for r in per_rank_summary
        }
        worst_rank = max(avg_total_by_rank, key=avg_total_by_rank.get)
        median_rank = _closest_rank_to_median(avg_total_by_rank)

        worst_avg_total_proxy_ms = avg_total_by_rank.get(worst_rank)
        median_avg_total_proxy_ms = (
            avg_total_by_rank.get(median_rank)
            if median_rank is not None
            else None
        )

        if (
            worst_avg_total_proxy_ms is not None
            and median_avg_total_proxy_ms is not None
            and median_avg_total_proxy_ms > 0.0
            and worst_rank is not None
            and median_rank is not None
            and worst_rank != median_rank
        ):
            worst_vs_median_pct = (
                100.0
                * (worst_avg_total_proxy_ms - median_avg_total_proxy_ms)
                / median_avg_total_proxy_ms
            )

    worst_summary = (
        per_rank_summary.get(worst_rank) if worst_rank is not None else None
    )
    median_summary = (
        per_rank_summary.get(median_rank) if median_rank is not None else None
    )

    worst_dl_share = (
        _share(
            worst_summary["avg_dataloader_ms"],
            worst_summary["avg_total_proxy_ms"],
        )
        if worst_summary
        else None
    )

    median_fwd_share = (
        _share(
            median_summary["avg_forward_ms"],
            median_summary["avg_total_proxy_ms"],
        )
        if median_summary
        else None
    )
    median_bwd_share = (
        _share(
            median_summary["avg_backward_ms"],
            median_summary["avg_total_proxy_ms"],
        )
        if median_summary
        else None
    )
    median_opt_share = (
        _share(
            median_summary["avg_optimizer_ms"],
            median_summary["avg_total_proxy_ms"],
        )
        if median_summary
        else None
    )

    worst_fwd_share = (
        _share(
            worst_summary["avg_forward_ms"],
            worst_summary["avg_total_proxy_ms"],
        )
        if worst_summary
        else None
    )
    worst_bwd_share = (
        _share(
            worst_summary["avg_backward_ms"],
            worst_summary["avg_total_proxy_ms"],
        )
        if worst_summary
        else None
    )
    worst_opt_share = (
        _share(
            worst_summary["avg_optimizer_ms"],
            worst_summary["avg_total_proxy_ms"],
        )
        if worst_summary
        else None
    )

    # Text card formatting
    w = 72
    sep = "-" * w

    lines: list[str] = []
    lines.append("")
    lines.append(sep)
    lines.append("STEP TIME")
    lines.append(sep)

    if not per_rank_summary:
        lines.append(f"training steps : {training_steps}")
        lines.append("ranks seen     : 0")
        lines.append("steps analyzed : n/a")
        lines.append("avg total step*: n/a")
        lines.append("dataloader     : n/a")
        lines.append("compute split  : n/a")
    else:
        lines.append(f"training steps : {training_steps}")
        lines.append(f"ranks seen     : {len(ranks_present)}")

        if worst_summary is not None:
            lines.append(
                f"steps analyzed : last {int(worst_summary['steps_analyzed']):,} / rank"
            )
        else:
            lines.append("steps analyzed : n/a")

        if (
            len(ranks_present) > 1
            and worst_rank is not None
            and median_rank is not None
        ):
            if worst_vs_median_pct is not None:
                lines.append(
                    f"straggler rank : r{worst_rank} (+{worst_vs_median_pct:.1f}% vs r{median_rank})"
                )
            else:
                lines.append(f"straggler rank : r{worst_rank}")
            lines.append(
                f"avg total step*: r{worst_rank} {_fmt_ms(worst_avg_total_proxy_ms)}"
                f" | r{median_rank} {_fmt_ms(median_avg_total_proxy_ms)}"
            )
            lines.append(
                f"dataloader     : worst r{worst_rank} {_fmt_pct(worst_dl_share)} of step*"
            )
            lines.append(
                f"median compute : fwd {_fmt_pct(median_fwd_share)}"
                f" | bwd {_fmt_pct(median_bwd_share)}"
                f" | opt {_fmt_pct(median_opt_share)}"
            )
            lines.append(
                f"worst compute  : fwd {_fmt_pct(worst_fwd_share)}"
                f" | bwd {_fmt_pct(worst_bwd_share)}"
                f" | opt {_fmt_pct(worst_opt_share)}"
            )
        else:
            only_rank = ranks_present[0]
            only_summary = per_rank_summary[only_rank]
            only_dl_share = _share(
                only_summary["avg_dataloader_ms"],
                only_summary["avg_total_proxy_ms"],
            )
            only_fwd_share = _share(
                only_summary["avg_forward_ms"],
                only_summary["avg_total_proxy_ms"],
            )
            only_bwd_share = _share(
                only_summary["avg_backward_ms"],
                only_summary["avg_total_proxy_ms"],
            )
            only_opt_share = _share(
                only_summary["avg_optimizer_ms"],
                only_summary["avg_total_proxy_ms"],
            )

            lines.append("straggler rank : n/a (single rank)")
            lines.append(
                f"avg total step*: r{only_rank} {_fmt_ms(only_summary['avg_total_proxy_ms'])}"
            )
            lines.append(
                f"dataloader     : r{only_rank} {_fmt_pct(only_dl_share)} of step*"
            )
            lines.append(
                f"compute split  : fwd {_fmt_pct(only_fwd_share)}"
                f" | bwd {_fmt_pct(only_bwd_share)}"
                f" | opt {_fmt_pct(only_opt_share)}"
            )

        lines.append("* total step = dataloader + max(step_cpu, gpu_compute)")
        lines.append("\n\n\n")

    card_text = "\n".join(lines).rstrip() + "\n"

    _append_text(db_path + "_summary_card.txt", card_text)

    existing = _load_json_or_empty(db_path + "_summary_card.json")
    existing["step_time"] = {
        "training_steps": training_steps,
        "latest_step_observed": latest_step_observed,
        "ranks_seen": len(ranks_present),
        "max_steps_analyzed_per_rank": int(max_rows),
        "worst_rank": worst_rank,
        "median_rank": median_rank,
        "worst_vs_median_pct": worst_vs_median_pct,
        "worst_avg_total_step_ms": worst_avg_total_proxy_ms,
        "median_avg_total_step_ms": median_avg_total_proxy_ms,
        "worst_dataloader_share_pct": worst_dl_share,
        "median_compute_share_pct": (
            {
                "forward": median_fwd_share,
                "backward": median_bwd_share,
                "optimizer": median_opt_share,
            }
            if median_summary is not None
            else None
        ),
        "worst_compute_share_pct": (
            {
                "forward": worst_fwd_share,
                "backward": worst_bwd_share,
                "optimizer": worst_opt_share,
            }
            if worst_summary is not None
            else None
        ),
        "per_rank": {
            str(rank): {
                **summary,
                "latest_step": latest_step_by_rank.get(rank),
            }
            for rank, summary in per_rank_summary.items()
        },
        "notes": {
            "total_step_definition": (
                "total_step = dataloader + max(step_cpu, forward + backward + optimizer)"
            ),
            "comparison_mode": (
                f"per-rank averages over each rank's last up to {int(max_rows)} steps"
            ),
        },
    }
    _write_json(db_path + "_summary_card.json", existing)

    if print_to_stdout:
        print(card_text, end="")

    return existing["step_time"]
