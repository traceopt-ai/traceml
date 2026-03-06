import json
import sqlite3
from typing import Any, Dict, Optional

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


def _event_bucket(name: str) -> Optional[str]:
    """
    Map event names to canonical buckets. Keep it permissive.
    Returns one of: dataloader, forward, backward, optimizer, other
    """
    n = name.lower()
    if "data" in n or "dataloader" in n or "input" in n or "batch" in n:
        return "dataloader"
    if "forward" in n or n in {"fwd"}:
        return "forward"
    if "backward" in n or "bwd" in n or "grad" in n:
        return "backward"
    if "optim" in n or "optimizer" in n or "step" == n or "update" in n:
        return "optimizer"
    return None


def generate_step_time_summary_card(
    db_path: str,
    step_sampler_name: str = "StepTimeSampler",
    max_steps: int = 5000,
) -> Dict[str, Any]:
    """
    Appends a shareable StepTime summary card beneath the existing SYSTEM card.

    Writes/updates:
      - <db_path>_summary_card.txt   (APPEND)
      - <db_path>_summary_card.json  (MERGE under key "step_time")
    """
    conn = sqlite3.connect(db_path)
    dec = msgspec.msgpack.Decoder(type=dict)

    cur = conn.execute(
        "SELECT payload_mp FROM raw_messages WHERE sampler = ? ORDER BY id ASC;",
        (step_sampler_name,),
    )

    steps_seen = 0

    # Step duration stats (ms)
    step_ms_sum = 0.0
    step_ms_max = 0.0

    # Canonical buckets (ms)
    bucket_sum: Dict[str, float] = {
        "dataloader": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
    }
    bucket_present: Dict[str, bool] = {k: False for k in bucket_sum}

    # For "top contributors" fallback
    event_total_ms: Dict[str, float] = {}

    for (blob,) in cur:
        if not blob:
            continue
        try:
            msg = dec.decode(blob)
        except Exception:
            continue

        tables = msg.get("tables")
        if not isinstance(tables, dict):
            continue

        for rows in tables.values():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if steps_seen >= max_steps:
                    break

                events = row.get("events")
                if not isinstance(events, dict) or not events:
                    continue

                # Compute total step time as sum of all event durations (ms)
                total_ms = 0.0

                for evt_name, by_dev in events.items():
                    if not isinstance(by_dev, dict):
                        continue

                    evt_ms = 0.0
                    for _dev, stats in by_dev.items():
                        if not isinstance(stats, dict):
                            continue
                        dur = stats.get("duration_ms")
                        if isinstance(dur, (int, float)):
                            evt_ms += float(dur)

                    if evt_ms <= 0.0:
                        continue

                    total_ms += evt_ms
                    name = str(evt_name)
                    event_total_ms[name] = (
                        event_total_ms.get(name, 0.0) + evt_ms
                    )

                    b = _event_bucket(name)
                    if b is not None:
                        bucket_sum[b] += evt_ms
                        bucket_present[b] = True

                if total_ms > 0.0:
                    steps_seen += 1
                    step_ms_sum += total_ms
                    step_ms_max = max(step_ms_max, total_ms)

        if steps_seen >= max_steps:
            break

    conn.close()

    avg_step_ms = (step_ms_sum / steps_seen) if steps_seen else None

    def fmt(x: Optional[float], suf: str = "", n_d: int = 1) -> str:
        return "n/a" if x is None else f"{x:.{n_d}f}{suf}"

    # Match SYSTEM card style: fixed-width separators + aligned layout
    w = 72
    sep = "-" * w

    lines: list[str] = []
    lines.append("")  # one clear spacer from the SYSTEM card
    lines.append(sep)
    lines.append("STEP TIME")
    lines.append(sep)

    if steps_seen == 0:
        # Keep it clean and non-alarming for early-cancelled runs
        lines.append("steps        : 0")
        lines.append("step avg/peak : n/a / n/a")
        lines.append("breakdown     : n/a")
    else:
        lines.append(f"steps        : {steps_seen}")
        lines.append(
            f"step avg/peak : {fmt(avg_step_ms,'ms',1)} / {fmt(step_ms_max,'ms',1)}"
        )

        # Prefer canonical buckets if we saw any of them; else show top events.
        if any(bucket_present.values()):
            denom = step_ms_sum if step_ms_sum > 0 else 1.0
            parts = []
            for k in ["dataloader", "forward", "backward", "optimizer"]:
                if bucket_present[k]:
                    share = 100.0 * (bucket_sum[k] / denom)
                    parts.append(f"{k} {share:.0f}%")
            lines.append(
                "breakdown     : " + (" | ".join(parts) if parts else "n/a")
            )
        else:
            top = sorted(
                event_total_ms.items(), key=lambda x: x[1], reverse=True
            )[:3]
            if top and step_ms_sum > 0:
                parts = [
                    f"{name} {100.0*(ms/step_ms_sum):.0f}%" for name, ms in top
                ]
                lines.append("top events    : " + " | ".join(parts))
            else:
                lines.append("top events    : n/a")

    card_text = "\n".join(lines).rstrip() + "\n"

    # Append to text card file (note: your system card uses _summary_card.txt)
    _append_text(db_path + "_summary_card.txt", card_text)

    # Merge JSON (keep your filename consistent: _summary_card.json)
    existing = _load_json_or_empty(db_path + "_summary_card.json")
    existing["step_time"] = {
        "steps": steps_seen,
        "step_avg_ms": avg_step_ms,
        "step_peak_ms": step_ms_max if steps_seen else None,
        "bucket_share_pct": (
            {
                k: (
                    (100.0 * bucket_sum[k] / step_ms_sum)
                    if (step_ms_sum > 0 and bucket_present[k])
                    else None
                )
                for k in bucket_sum
            }
            if steps_seen
            else {}
        ),
        "top_events_share_pct": (
            None
            if any(bucket_present.values())
            else (
                [
                    {"event": name, "share_pct": 100.0 * (ms / step_ms_sum)}
                    for name, ms in sorted(
                        event_total_ms.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:3]
                ]
                if step_ms_sum > 0
                else []
            )
        ),
    }
    _write_json(db_path + "_summary_card.json", existing)

    return existing["step_time"]
