from typing import Dict, Any, Optional
from traceml.database.database import Database


class LayerCombinedTimerData:
    """
    Computes per-layer timing stats from Time DB.

    Cache format per layer:
        {
            "current": float (ms),
            "avg": float (ms),
            "on_gpu": bool,
        }
    """

    def __init__(
        self,
        forward_db: Optional[Database],
        backward_db: Optional[Database],
        top_n_layers: Optional[int] = 20,
    ):
        self._forward_db = forward_db
        self._backward_db = backward_db
        self._top_n = top_n_layers

        self._forward_cache: Dict[str, Dict[str, Any]] = {}
        self._backward_cache: Dict[str, Dict[str, Any]] = {}

    def compute_display_data(self) -> Dict[str, Any]:
        act_snapshot = self._compute_snapshot(is_forward=True)
        grad_snapshot = self._compute_snapshot(is_forward=False)

        self._merge_cache(self._forward_cache, act_snapshot)
        self._merge_cache(self._backward_cache, grad_snapshot)

        layers = set(self._forward_cache.keys()) | set(self._backward_cache.keys())
        if not layers:
            return {
                "top_items": [],
                "all_items": [],
                "other": {
                    "total_forward_current": 0.0,
                    "total_forward_avg": 0.0,
                    "total_backward_current": 0.0,
                    "total_backward_avg": 0.0,
                    "pct": 0.0,
                },
            }

        # Build rows (NO summing of activation + gradient per layer)
        rows = []
        for layer in layers:
            act = self._forward_cache.get(layer, {})
            grad = self._backward_cache.get(layer, {})

            act_cur = float(act.get("current", 0.0))
            act_avg = float(act.get("avg", 0.0))
            act_on_gpu = act.get("on_gpu", None)

            grad_cur = float(grad.get("current", 0.0))
            grad_avg = float(grad.get("avg", 0.0))
            grad_on_gpu = grad.get("on_gpu", None)

            # Device: use activation if present, else gradient, else False
            if act_on_gpu is not None:
                on_gpu = bool(act_on_gpu)
            elif grad_on_gpu is not None:
                on_gpu = bool(grad_on_gpu)
            else:
                on_gpu = False

            rows.append(
                {
                    "layer": layer,
                    "forward_current": act_cur,
                    "forward_avg": act_avg,
                    "backward_current": grad_cur,
                    "backward_avg": grad_avg,
                    "on_gpu": on_gpu,
                }
            )

        total_current_sum_ms = sum(
            (r["forward_current"] + r["backward_current"]) for r in rows
        )
        for r in rows:
            layer_total = r["forward_current"] + r["backward_current"]
            r["pct"] = (
                (layer_total / total_current_sum_ms * 100.0)
                if total_current_sum_ms > 0
                else 0.0
            )

        ## Sorting based on sum of forward and backward avg (layer takes most time)
        def sort_key(r: Dict[str, Any]) -> float:
            return float(r["forward_avg"]) + float(r["backward_avg"])

        rows_sorted = sorted(rows, key=sort_key, reverse=True)
        top_items = rows_sorted[: self._top_n]
        other_items = rows_sorted[self._top_n :]

        other_act_cur_sum = sum(r["forward_current"] for r in other_items)
        other_grad_cur_sum = sum(r["backward_current"] for r in other_items)

        other = {
            "total_forward_current": other_act_cur_sum,
            "total_forward_avg": sum(r["forward_avg"] for r in other_items),
            "total_backward_current": other_grad_cur_sum,
            "total_backward_avg": sum(r["backward_avg"] for r in other_items),
            "pct": (
                ((other_act_cur_sum + other_grad_cur_sum) / total_current_sum_ms)
                * 100.0
                if total_current_sum_ms > 0
                else 0.0
            ),
        }
        return {
            "top_items": top_items,
            "all_items": rows_sorted,
            "other": other,
            "activation_cache": self._forward_cache,
            "gradient_cache": self._backward_cache,
        }

    def _compute_snapshot(self, is_forward: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Reads DB tables and produces a one-pass snapshot.
        """
        db = self._forward_db if is_forward else self._backward_db
        snapshot = {}

        for layer, rows in db.all_tables().items():
            if not rows:
                continue

            last = rows[-1]
            on_gpu = bool(last.get("on_gpu", False))

            cur = float(
                (last.get("gpu_duration_ms") if on_gpu else last.get("cpu_duration_ms"))
                or 0.0
            )
            durations = []
            for r in rows:
                d = (
                    r.get("gpu_duration_ms")
                    if r.get("on_gpu")
                    else r.get("cpu_duration_ms")
                )
                if d is not None:
                    durations.append(float(d))
            avg = (sum(durations) / len(durations)) if durations else cur
            snapshot[layer] = {
                "current": cur,
                "avg": avg,
                "on_gpu": on_gpu,
            }
        return snapshot

    @staticmethod
    def _merge_cache(
        cache: Dict[str, Dict[str, Any]],
        snapshot: Dict[str, Dict[str, Any]],
    ) -> None:
        """
        Merge snapshot into cache (current overwritten, global is max).
        """
        for layer, entry in snapshot.items():
            if layer not in cache:
                cache[layer] = entry
            else:
                cache[layer]["current"] = entry["current"]
                cache[layer]["avg"] = cache[layer]["avg"] * 0.9 + entry["avg"] * 0.1   # EMA
                cache[layer]["on_gpu"] = entry.get(
                    "on_gpu", cache[layer].get("on_gpu", False)
                )


class LayerCombinedTimerSummary:
    """
    Computes global statistics for log_summary():
      - total samples (forward + backward events)
      - total layers seen
      - avg / p50 / p95 forward time
      - avg / p50 / p95 backward time
    """

    def __init__(
        self,
        forward_db: Optional[Database],
        backward_db: Optional[Database] = None,
    ):
        self._forward_db = forward_db
        self._backward_db = backward_db

    def compute_layer_timing_summary(self) -> Dict[str, Any]:
        fwd = self._compute_db_summary(self._forward_db)
        bwd = self._compute_db_summary(self._backward_db)

        total_samples = max(fwd["total_samples"], bwd["total_samples"])
        total_layers_seen = len(set(fwd["layers_seen"]) | set(bwd["layers_seen"]))

        return {
            "total_samples": total_samples,
            "total_layers_seen": total_layers_seen,
            # Forward
            "avg_forward_ms": fwd["average"],
            "p50_forward_ms": fwd["p50_ms"],
            "p95_forward_ms": fwd["p95_ms"],
            # Backward
            "avg_backward_ms": bwd["average"],
            "p50_backward_ms": bwd["p50_ms"],
            "p95_backward_ms": bwd["p95_ms"],
        }

    def _compute_db_summary(self, db: Optional[Database]) -> Dict[str, Any]:
        if db is None:
            return {
                "total_samples": 0,
                "layers_seen": set(),
                "average": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
            }

        layers_seen = set()
        durations = []

        for layer_name, rows in db.all_tables().items():
            if not rows:
                continue
            layers_seen.add(layer_name)
            for r in rows:
                d = self._pick_duration_ms(r)
                if d is not None:
                    durations.append(float(d))

        total_samples = len(durations)
        if total_samples:
            durations_sorted = sorted(durations)
            average = sum(durations_sorted) / total_samples
            p50 = durations_sorted[int(0.50 * (total_samples - 1))]
            p95 = durations_sorted[int(0.95 * (total_samples - 1))]
        else:
            average = p50 = p95 = 0.0


        return {
            "total_samples": total_samples,
            "layers_seen": layers_seen,
            "average": average,
            "p50_ms": p50,
            "p95_ms": p95,
        }

    @staticmethod
    def _pick_duration_ms(row: Dict[str, Any]) -> Optional[float]:
        """
        Prefer GPU duration when on_gpu else CPU duration.
        Returns None if missing.
        """
        on_gpu = bool(row.get("on_gpu", False))
        d = row.get("gpu_duration_ms") if on_gpu else row.get("cpu_duration_ms")
        if d is None:
            return None
        return float(d)

    def compute_global_averages(self, is_forward: bool) -> Dict[str, float]:
        db = self._forward_db if is_forward else self._backward_db
        avgs: Dict[str, float] = {}

        if db is None:
            return avgs

        for layer_name, rows in db.all_tables().items():
            durations = []
            for r in rows:
                d = self._pick_duration_ms(r)
                if d is not None:
                    durations.append(float(d))
            avgs[layer_name] = (
                sum(durations) / len(durations) if durations else 0.0
            )

        return avgs

    @staticmethod
    def top_n_from_dict(d: Dict[str, float], n: int = 3):
        if not d:
            return []
        return sorted(d.items(), key=lambda kv: float(kv[1]), reverse=True)[:n]
