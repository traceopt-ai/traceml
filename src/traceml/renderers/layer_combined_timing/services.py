from typing import Dict, Any, Optional
from traceml.database.database import Database


class LayerTimingData:
    """
    Computes per-layer timing stats from ActivationTime DB.

    Cache format per layer:
        {
            "current": float (ms),
            "global": float (ms),
            "on_gpu": bool,
        }
    """

    def __init__(
        self,
        timing_db: Database,
        top_n_layers: Optional[int] = 20,
    ):
        self._db = timing_db
        self._top_n = top_n_layers
        self._cache: Dict[str, Dict[str, Any]] = {}

    def compute_display_data(self) -> Dict[str, Any]:
        snapshot = self._compute_snapshot()
        self._merge_cache(snapshot)

        if not self._cache:
            return {
                "top_items": [],
                "other": {"current": 0.0, "global": 0.0, "pct": 0.0},
                "total_current_sum": 0.0,
            }

        # ---- sorting by global peak ----
        sorted_items = sorted(
            self._cache.items(),
            key=lambda kv: kv[1]["global"],
            reverse=True,
        )

        top_items = sorted_items[: self._top_n]
        other_items = sorted_items[self._top_n :]

        total_current_sum = sum(v["current"] for v in self._cache.values())

        rows = []
        for layer, entry in top_items:
            pct = (
                (entry["current"] / total_current_sum) * 100.0
                if total_current_sum > 0
                else 0.0
            )
            rows.append(
                {
                    "layer": layer,
                    "current": entry["current"],
                    "global": entry["global"],
                    "on_gpu": entry["on_gpu"],
                    "pct": pct,
                }
            )

        other_current = sum(v["current"] for _, v in other_items)
        other_global = max((v["global"] for _, v in other_items), default=0.0)
        other_pct = (
            (other_current / total_current_sum) * 100.0
            if total_current_sum > 0
            else 0.0
        )

        return {
            "top_items": rows,
            "other": {
                "current": other_current,
                "global": other_global,
                "pct": other_pct,
            },
            "total_current_sum": total_current_sum,
        }

    # --------------------------------------------------

    def _compute_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """
        Reads DB tables and produces a one-pass snapshot.
        """
        snapshot = {}

        for layer, rows in self._db.all_tables().items():
            if not rows:
                continue

            last = rows[-1]
            on_gpu = bool(last.get("on_gpu", False))

            if on_gpu:
                cur = float(last.get("gpu_duration_ms", 0.0) or 0.0)
            else:
                cur = float(last.get("cpu_duration_ms", 0.0) or 0.0)

            peak = cur
            for r in rows:
                d = (
                    r.get("gpu_duration_ms")
                    if r.get("on_gpu")
                    else r.get("cpu_duration_ms")
                )
                if d is not None:
                    peak = max(peak, float(d))

            snapshot[layer] = {
                "current": cur,
                "global": peak,
                "on_gpu": on_gpu,
            }

        return snapshot

    def _merge_cache(self, snapshot: Dict[str, Dict[str, Any]]) -> None:
        """
        Merge snapshot into global cache.
        """
        for layer, entry in snapshot.items():
            if layer not in self._cache:
                self._cache[layer] = entry
            else:
                self._cache[layer]["current"] = entry["current"]
                self._cache[layer]["global"] = max(
                    self._cache[layer]["global"], entry["global"]
                )
