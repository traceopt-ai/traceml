from typing import Dict, Any, Optional
from traceml.database.database import Database


class LayerCombinedMemoryData:

    def __init__(
        self,
        layer_table,
        activation_db: Database,
        gradient_db: Database,
        top_n_layers: Optional[int] = 20,
    ):
        self._layer_table = layer_table
        self._activation_db = activation_db
        self._gradient_db = gradient_db
        self._top_n = top_n_layers

        # caches store cumulative peaks
        self._activation_cache: Dict[str, Dict[str, float]] = {}
        self._gradient_cache: Dict[str, Dict[str, float]] = {}

    def compute_display_data(self) -> Dict[str, Any]:
        """
        Returns ALL per-layer data needed by dashboard, CLI, notebook.

        Sorting = total_peak_memory (param + activation_peak + grad_peak)
        Percent (%) = total_current_memory / sum(total_current_memory)
        """

        # Load table snapshot
        layer_snapshot = self._compute_layer_snapshot()
        act_snapshot = self._compute_snapshot(is_activation=True)
        grad_snapshot = self._compute_snapshot(is_activation=False)

        # Update global caches
        self._merge_cache(self._activation_cache, act_snapshot)
        self._merge_cache(self._gradient_cache, grad_snapshot)

        layers = layer_snapshot.get("layer_memory", {})  # param memory per layer
        model_index = layer_snapshot.get("model_index")

        peak_map = {}
        current_map = {}

        # Compute peak & current totals (single pass)
        for layer, param_mem in layers.items():
            act_cur = self._activation_cache.get(layer, {}).get("current", 0.0)
            act_peak = self._activation_cache.get(layer, {}).get("global", 0.0)

            grad_cur = self._gradient_cache.get(layer, {}).get("current", 0.0)
            grad_peak = self._gradient_cache.get(layer, {}).get("global", 0.0)

            peak_map[layer] = float(param_mem) + float(act_peak) + float(grad_peak)
            current_map[layer] = float(param_mem) + float(act_cur) + float(grad_cur)

        total_current_sum = sum(current_map.values()) if current_map else 0.0

        all_rows = [
            self._build_layer_row(
                layer=layer,
                param_mem=layers.get(layer, 0.0),
                current_map=current_map,
                peak_map=peak_map,
                total_current_sum=total_current_sum,
            )
            for layer in layers.keys()
        ]

        # Sort by total peak memory
        all_rows_sorted = sorted(
            all_rows,
            key=lambda r: r["total_peak_memory"],
            reverse=True,
        )

        #  Split top / other
        top_items = all_rows_sorted[: self._top_n]
        other_items = all_rows_sorted[self._top_n:]

        #  Aggregate "other"
        other_current_total = sum(
            r["total_current_memory"] for r in other_items
        ) if other_items else 0.0

        other = {
            "param_memory": sum(r["param_memory"] for r in other_items),
            "activation_current": sum(r["activation_current"] for r in other_items),
            "activation_peak": sum(r["activation_peak"] for r in other_items),
            "gradient_current": sum(r["gradient_current"] for r in other_items),
            "gradient_peak": sum(r["gradient_peak"] for r in other_items),
            "total_current_memory": other_current_total,
            "pct": (
                other_current_total / total_current_sum * 100.0
                if total_current_sum else 0.0
            ),
        }

        return {
            "model_index": model_index,

            "top_items": top_items,
            "other": other,
            "all_items": all_rows_sorted,

            "total_current_sum": total_current_sum,
            "total_peak_sum": sum(peak_map.values()),
        }

    def _build_layer_row(
            self,
            layer: str,
            param_mem: float,
            current_map: Dict[str, float],
            peak_map: Dict[str, float],
            total_current_sum: float,
    ) -> Dict[str, Any]:

        act_cur = self._activation_cache.get(layer, {}).get("current", 0.0)
        act_peak = self._activation_cache.get(layer, {}).get("global", 0.0)
        grad_cur = self._gradient_cache.get(layer, {}).get("current", 0.0)
        grad_peak = self._gradient_cache.get(layer, {}).get("global", 0.0)

        current_total = current_map[layer]
        pct = (current_total / total_current_sum * 100.0) if total_current_sum else 0.0

        return {
            "layer": layer,

            "param_memory": float(param_mem),
            "activation_current": float(act_cur),
            "activation_peak": float(act_peak),
            "gradient_current": float(grad_cur),
            "gradient_peak": float(grad_peak),

            "total_peak_memory": float(peak_map[layer]),
            "total_current_memory": float(current_total),
            "pct": pct,
        }

    def _compute_layer_snapshot(self) -> Dict[str, Any]:
        if not self._layer_table:
            return {"layer_memory": {}, "model_index": "—"}

        last = self._layer_table[-1]
        return {
            "layer_memory": last.get("layer_memory", {}) or {},
            "model_index": last.get("model_index", "—"),
        }

    def _compute_snapshot(self, is_activation: bool) -> Dict[str, Dict[str, float]]:
        db = self._activation_db if is_activation else self._gradient_db
        layer_peaks = {}
        layer_current = {}

        for layer, rows in db.all_tables().items():
            if not rows:
                continue

            latest_per_device = {}
            global_peak = 0.0

            for r in rows:
                mem = r.get("memory", {}) or {}
                for dev, size in mem.items():
                    size_f = float(size)
                    latest_per_device[dev] = size_f
                    global_peak = max(global_peak, size_f)

            layer_current[layer] = max(latest_per_device.values()) if latest_per_device else 0.0
            layer_peaks[layer] = global_peak

        return {
            layer: {
                "current_peak": layer_current.get(layer, 0.0),
                "global_peak": layer_peaks.get(layer, 0.0),
            }
            for layer in (set(layer_current) | set(layer_peaks))
        }

    def _merge_cache(self, cache, new_data):
        if not new_data:
            return
        for layer, entry in new_data.items():
            cur = entry.get("current_peak", 0.0)
            gbl = entry.get("global_peak", 0.0)
            if layer not in cache:
                cache[layer] = {"current": cur, "global": gbl}
            else:
                cache[layer]["current"] = cur
                cache[layer]["global"] = max(cache[layer]["global"], gbl)


class LayerCombinedMemorySummary:
    """
    Computes global statistics for log_summary():
      - total samples
      - #models
      - avg & peak memory
      - global activation / gradient peaks per layer
    """

    def __init__(
        self,
        layer_table,
        activation_db: Database,
        gradient_db: Database,
    ):
        self._layer_table = layer_table
        self._activation_db = activation_db
        self._gradient_db = gradient_db

    def compute_layer_memory_summary(self) -> Dict[str, Any]:
        if not self._layer_table:
            return {
                "total_models_seen": 0,
                "model_memory": 0.0,
            }

        total_samples = len(self._layer_table)
        model_signatures = {
            entry.get("model_signature") for entry in self._layer_table
        }

        totals = [
            float(entry.get("total_memory", 0.0)) for entry in self._layer_table
        ]
        avg_memory = sum(totals) / len(totals) if totals else 0.0

        return {
            "total_models_seen": len(model_signatures),
            "model_memory": avg_memory,
        }

    # ------------------------------------------------------------------
    # Global peaks (for top-k lists)
    # ------------------------------------------------------------------

    def compute_global_peaks(self, is_activation: bool) -> Dict[str, float]:
        """
        Compute global peak per layer from activation/gradient_db.
        Equivalent to previous _compute_peaks.
        """
        db = self._activation_db if is_activation else self._gradient_db

        peaks: Dict[str, float] = {}
        for layer_name, rows in db.all_tables().items():
            peak = 0.0
            for r in rows:
                mem = r.get("memory", {}) or {}
                if mem:
                    peak = max(peak, max(float(v) for v in mem.values()))
            peaks[layer_name] = peak
        return peaks

    # ------------------------------------------------------------------
    # Helper: top-n from dict
    # ------------------------------------------------------------------

    @staticmethod
    def top_n_from_dict(d: Dict[str, float], n: int = 3):
        if not d:
            return []
        return sorted(d.items(), key=lambda kv: float(kv[1]), reverse=True)[:n]
