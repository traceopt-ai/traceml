from typing import Dict, Any, Optional
from traceml.database.database import Database


class LayerCombinedTimerData:
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
        activation_db: Optional[Database],
        gradient_db: Optional[Database],
        top_n_layers: Optional[int] = 20,
    ):
        self._activation_db = activation_db
        self._gradient_db = gradient_db
        self._top_n = top_n_layers

        self._activation_cache: Dict[str, Dict[str, Any]] = {}
        self._gradient_cache: Dict[str, Dict[str, Any]] = {}

    def compute_display_data(self) -> Dict[str, Any]:
        act_snapshot = self._compute_snapshot(is_activation=True)
        grad_snapshot = self._compute_snapshot(is_activation=False)

        self._merge_cache(self._activation_cache, act_snapshot)
        self._merge_cache(self._gradient_cache, grad_snapshot)

        layers = set(self._activation_cache.keys()) | set(self._gradient_cache.keys())
        if not layers:
            return {
                "top_items": [],
                "all_items": [],
                "other": {
                    "activation_current_sum_ms": 0.0,
                    "activation_peak_max_ms": 0.0,
                    "gradient_current_sum_ms": 0.0,
                    "gradient_peak_max_ms": 0.0,
                    "pct": 0.0,
                },
            }

        # Build rows (NO summing of activation + gradient per layer)
        rows = []
        for layer in layers:
            act = self._activation_cache.get(layer, {})
            grad = self._gradient_cache.get(layer, {})

            act_cur = float(act.get("current", 0.0))
            act_peak = float(act.get("global", 0.0))
            act_on_gpu = act.get("on_gpu", None)

            grad_cur = float(grad.get("current", 0.0))
            grad_peak = float(grad.get("global", 0.0))
            grad_on_gpu = grad.get("on_gpu", None)

            # Device: use activation if present, else gradient, else False
            if act_on_gpu is not None:
                on_gpu = bool(act_on_gpu)
            elif grad_on_gpu is not None:
                on_gpu = bool(grad_on_gpu)
            else:
                on_gpu = False

            rows.append({
                "layer": layer,
                "activation_current_ms": act_cur,
                "activation_peak_ms": act_peak,
                "gradient_current_ms": grad_cur,
                "gradient_peak_ms": grad_peak,
                "on_gpu": on_gpu,
            })

        total_current_sum_ms = sum(
            (r["activation_current_ms"] + r["gradient_current_ms"]) for r in rows
        )
        for r in rows:
            layer_total = r["activation_current_ms"] + r["gradient_current_ms"]
            r["pct"] = (layer_total / total_current_sum_ms * 100.0) if total_current_sum_ms > 0 else 0.0

        ## Sorting based on sum of activation and gradient peak (layer takes most memory)
        def sort_key(r: Dict[str, Any]) -> float:
            return max(float(r["activation_peak_ms"]), float(r["gradient_peak_ms"]))

        rows_sorted = sorted(rows, key=sort_key, reverse=True)
        top_items = rows_sorted[: self._top_n]
        other_items = rows_sorted[self._top_n :]

        other_act_cur_sum = sum(r["activation_current_ms"] for r in other_items)
        other_grad_cur_sum = sum(r["gradient_current_ms"] for r in other_items)

        other = {
            "activation_current_sum_ms": other_act_cur_sum,
            "activation_peak_max_ms": sum(r["activation_peak_ms"] for r in other_items),
            "gradient_current_sum_ms": other_grad_cur_sum,
            "gradient_peak_max_ms": sum(r["gradient_peak_ms"] for r in other_items),
            "pct": (
                ((other_act_cur_sum + other_grad_cur_sum) / total_current_sum_ms) * 100.0
                if total_current_sum_ms > 0 else 0.0
            ),
        }
        return {
            "top_items": top_items,
            "all_items": rows_sorted,
            "other": other,
            "activation_cache": self._activation_cache,
            "gradient_cache": self._gradient_cache,
        }


    def _compute_snapshot(
        self, is_activation:bool=True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Reads DB tables and produces a one-pass snapshot.
        """
        db = self._activation_db if is_activation else self._gradient_db
        snapshot = {}

        for layer, rows in db.all_tables().items():
            if not rows:
                continue

            last = rows[-1]
            on_gpu = bool(last.get("on_gpu", False))

            cur = float(
                (last.get("gpu_duration_ms") if on_gpu else last.get("cpu_duration_ms")) or 0.0
            )
            peak = cur
            for r in rows:
                d = (r.get("gpu_duration_ms") if r.get("on_gpu") else r.get("cpu_duration_ms"))
                if d is not None:
                    peak = max(peak, float(d))
            snapshot[layer] = {
                "current": cur,
                "global": peak,
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
                cache[layer]["global"] = max(cache[layer]["global"], entry["global"])
                cache[layer]["on_gpu"] = entry.get("on_gpu", cache[layer].get("on_gpu", False))


class LayerCombinedTimerSummary:
    """
    Computes global statistics for log_summary():
      - total samples (activation + gradient events)
      - total layers seen
      - avg & peak activation time
      - avg & peak gradient time
      - global activation / gradient peaks per layer
    """

    def __init__(
        self,
        activation_db: Optional[Database],
        gradient_db: Optional[Database] = None,
    ):
        self._activation_db = activation_db
        self._gradient_db = gradient_db


    def compute_layer_timing_summary(self) -> Dict[str, Any]:
        act = self._compute_db_summary(self._activation_db)
        grad = self._compute_db_summary(self._gradient_db)

        total_samples = max(act["total_samples"], grad["total_samples"])
        total_layers_seen = len(set(act["layers_seen"]) | set(grad["layers_seen"]))

        return {
            "total_samples": total_samples,
            "total_layers_seen": total_layers_seen,

            "average_activation_time_ms": act["average_ms"],
            "peak_activation_time_ms": act["peak_ms"],

            "average_gradient_time_ms": grad["average_ms"],
            "peak_gradient_time_ms": grad["peak_ms"],
        }

    def _compute_db_summary(self, db: Optional[Database]) -> Dict[str, Any]:
        if db is None:
            return {
                "total_samples": 0,
                "layers_seen": set(),
                "average_ms": 0.0,
                "peak_ms": 0.0,
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
        average_ms = (sum(durations) / total_samples) if total_samples else 0.0
        peak_ms = max(durations) if durations else 0.0

        return {
            "total_samples": total_samples,
            "layers_seen": layers_seen,
            "average_ms": average_ms,
            "peak_ms": peak_ms,
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


    def compute_global_peaks(self, is_activation: bool) -> Dict[str, float]:
        db = self._activation_db if is_activation else self._gradient_db
        peaks: Dict[str, float] = {}

        if db is None:
            return peaks

        for layer_name, rows in db.all_tables().items():
            peak = 0.0
            for r in rows:
                d = self._pick_duration_ms(r)
                if d is not None:
                    peak = max(peak, float(d))
            peaks[layer_name] = peak

        return peaks

    @staticmethod
    def top_n_from_dict(d: Dict[str, float], n: int = 3):
        if not d:
            return []
        return sorted(d.items(), key=lambda kv: float(kv[1]), reverse=True)[:n]
