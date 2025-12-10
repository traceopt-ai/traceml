from typing import Dict, Any, Optional
from traceml.database.database import Database


class LayerCombinedData:

    def __init__(
        self,
        layer_table,
        activation_db: Database,
        gradient_db: Database,
        top_n_layers: Optional[int] = 10,
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

        # ---- Load table snapshot & memory DBs ----
        layer_snapshot = self._compute_layer_snapshot()
        act_snapshot = self._compute_snapshot(is_activation=True)
        grad_snapshot = self._compute_snapshot(is_activation=False)

        # Update global caches
        self._merge_cache(self._activation_cache, act_snapshot)
        self._merge_cache(self._gradient_cache, grad_snapshot)

        layers = layer_snapshot["layer_memory"]         # parameter memory
        model_index = layer_snapshot["model_index"]
        activation_cache = self._activation_cache
        gradient_cache = self._gradient_cache

        peak_map = {}
        current_map = {}

        for layer, param_mem in layers.items():

            act_cur  = activation_cache.get(layer, {}).get("current", 0.0)
            act_peak = activation_cache.get(layer, {}).get("global", 0.0)

            grad_cur  = gradient_cache.get(layer, {}).get("current", 0.0)
            grad_peak = gradient_cache.get(layer, {}).get("global", 0.0)

            # Peak memory = used for sorting (static layer cost)
            peak_map[layer] = float(param_mem) + float(act_peak) + float(grad_peak)

            # Current memory = used for percentage display
            current_map[layer] = float(param_mem) + float(act_cur) + float(grad_cur)

        sorted_items = sorted(
            peak_map.items(), key=lambda kv: float(kv[1]), reverse=True
        )
        top_items = sorted_items[: self._top_n]
        other_items = sorted_items[self._top_n:]

        total_current_sum = sum(current_map.values()) if current_map else 0.0

        # ---- Build top rows ----
        rows = []
        for layer, peak_val in top_items:

            param_mem = layers.get(layer, 0.0)
            act_cur  = activation_cache.get(layer, {}).get("current", 0.0)
            act_peak = activation_cache.get(layer, {}).get("global", 0.0)
            grad_cur  = gradient_cache.get(layer, {}).get("current", 0.0)
            grad_peak = gradient_cache.get(layer, {}).get("global", 0.0)

            current_total = current_map[layer]
            pct = (current_total / total_current_sum * 100.0) if total_current_sum else 0.0

            rows.append({
                "layer": layer,

                "param_memory": param_mem,
                "activation_current": act_cur,
                "activation_peak": act_peak,
                "gradient_current": grad_cur,
                "gradient_peak": grad_peak,

                "total_peak_memory": peak_val,       # used for sorting
                "total_current_memory": current_total,  # used for %
                "pct": pct,
            })

        other_layers = [layer for layer, _ in other_items]

        other_param_sum = sum(layers.get(l, 0.0) for l in other_layers)
        other_act_cur = sum(activation_cache.get(l, {}).get("current", 0.0) for l in other_layers)
        other_act_peak = sum(activation_cache.get(l, {}).get("global", 0.0) for l in other_layers)
        other_grad_cur = sum(gradient_cache.get(l, {}).get("current", 0.0) for l in other_layers)
        other_grad_peak = sum(gradient_cache.get(l, {}).get("global", 0.0) for l in other_layers)

        other_current_total = sum(current_map[l] for l in other_layers) if other_layers else 0.0
        other_pct = (other_current_total / total_current_sum * 100.0) if total_current_sum else 0.0

        return {
            "model_index": model_index,

            # per-layer rows
            "top_items": rows,

            # aggregated "other" row
            "other": {
                "param_memory": other_param_sum,
                "activation_current": other_act_cur,
                "activation_peak": other_act_peak,
                "gradient_current": other_grad_cur,
                "gradient_peak": other_grad_peak,
                "total_current_memory": other_current_total,
                "pct": other_pct,
            },

            # sums for final footer
            "total_current_sum": total_current_sum,
            "total_peak_sum": sum(peak_map.values()),

            "activation_cache": activation_cache,
            "gradient_cache": gradient_cache,
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


class LayerMemorySummary:
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
                "total_samples": 0,
                "total_models_seen": 0,
                "average_model_memory": 0.0,
                "peak_model_memory": 0.0,
            }

        total_samples = len(self._layer_table)
        model_signatures = {
            entry.get("model_signature") for entry in self._layer_table
        }

        totals = [
            float(entry.get("total_memory", 0.0)) for entry in self._layer_table
        ]
        avg_memory = sum(totals) / len(totals) if totals else 0.0
        peak_memory = max(totals) if totals else 0.0

        return {
            "total_samples": total_samples,
            "total_models_seen": len(model_signatures),
            "average_model_memory": avg_memory,
            "peak_model_memory": peak_memory,
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
