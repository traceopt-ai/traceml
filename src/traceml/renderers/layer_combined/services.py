from typing import Dict, Any, Optional

from traceml.database.database import Database


class LayerCombinedData:
    """
    Computes the data needed by LayerCombinedRenderer for:
      - top-N layer memory rows
      - "Other Layers" aggregate row
      - activation & gradient peak caches

    This is the main “data model” for the combined renderer.
    """

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

        # caches: {layer: {"current": float, "global": float}}
        self._activation_cache: Dict[str, Dict[str, float]] = {}
        self._gradient_cache: Dict[str, Dict[str, float]] = {}

    def compute_display_data(self) -> Dict[str, Any]:
        """
        Core entry for renderer: returns everything needed
        to render CLI / HTML tables.

        Structure:
        {
            "top_items": [(layer_name, memory_bytes), ...],
            "other": {
                "total": float,
                "pct": float,
                "activation": {"current": float, "global": float},
                "gradient": {"current": float, "global": float},
            },
            "total_memory": float,
            "model_index": Any,
            "activation_cache": {...},
            "gradient_cache": {...},
        }
        """
        layer_snapshot = self._compute_layer_snapshot()
        act_snapshot = self._compute_snapshot(is_activation=True)
        grad_snapshot = self._compute_snapshot(is_activation=False)

        # update caches
        self._merge_cache(self._activation_cache, act_snapshot)
        self._merge_cache(self._gradient_cache, grad_snapshot)

        layers = layer_snapshot["layer_memory"]
        total_memory = layer_snapshot["total_memory"]
        model_index = layer_snapshot["model_index"]

        sorted_items = sorted(
            layers.items(), key=lambda kv: float(kv[1]), reverse=True
        )
        top_items = sorted_items[: self._top_n] if self._top_n else sorted_items
        other_items = sorted_items[self._top_n :] if self._top_n else []

        other_total = sum(float(v) for _, v in other_items)
        pct_other = (other_total / total_memory * 100.0) if total_memory > 0.0 else 0.0

        def _agg_peaks(
            cache: Dict[str, Dict[str, float]], names: list[str]
        ) -> Dict[str, float]:
            if not names:
                return {"current": 0.0, "global": 0.0}
            cur = sum(cache.get(n, {}).get("current", 0.0) for n in names)
            gbl = sum(cache.get(n, {}).get("global", 0.0) for n in names)
            return {"current": cur, "global": gbl}

        other_act = _agg_peaks(self._activation_cache, [n for n, _ in other_items])
        other_grad = _agg_peaks(self._gradient_cache, [n for n, _ in other_items])

        return {
            "top_items": top_items,
            "other": {
                "total": other_total,
                "pct": pct_other,
                "activation": other_act,
                "gradient": other_grad,
            },
            "total_memory": total_memory,
            "model_index": model_index,
            "activation_cache": self._activation_cache,
            "gradient_cache": self._gradient_cache,
        }

    # ------------------------------------------------------------------
    # Internal: layer / activation / gradient snapshots
    # ------------------------------------------------------------------

    def _compute_layer_snapshot(self) -> Dict[str, Any]:
        """
        Return last entry from layer_table safely.
        """
        if not self._layer_table:
            return {
                "layer_memory": {},
                "total_memory": 0.0,
                "model_index": "—",
            }

        last = self._layer_table[-1]
        return {
            "layer_memory": last.get("layer_memory", {}) or {},
            "total_memory": last.get("total_memory", 0.0) or 0.0,
            "model_index": last.get("model_index", "—"),
        }

    def _compute_snapshot(self, is_activation: bool) -> Dict[str, Dict[str, float]]:
        """
        Return CURRENT and PEAK memory per layer for activation/gradient DBs.

        Result:
        {
            layer_name: {
                "current_peak": float,
                "global_peak": float,
            },
            ...
        }
        """
        db = self._activation_db if is_activation else self._gradient_db
        layer_peaks: Dict[str, float] = {}
        layer_current: Dict[str, float] = {}

        for layer, rows in db.all_tables().items():
            if not rows:
                continue

            latest_per_device: Dict[str, float] = {}
            global_peak = 0.0

            for r in rows:
                mem = r.get("memory", {}) or {}
                for dev, size in mem.items():
                    size_f = float(size)
                    latest_per_device[dev] = size_f
                    global_peak = max(global_peak, size_f)

            current_peak = (
                max(latest_per_device.values()) if latest_per_device else 0.0
            )
            layer_current[layer] = current_peak
            layer_peaks[layer] = global_peak

        merged = {
            layer: {
                "current_peak": layer_current.get(layer, 0.0),
                "global_peak": layer_peaks.get(layer, 0.0),
            }
            for layer in (set(layer_current) | set(layer_peaks))
        }
        return merged

    # ------------------------------------------------------------------
    # Cache merging logic
    # ------------------------------------------------------------------

    def _merge_cache(
        self,
        cache: Dict[str, Dict[str, float]],
        new_data: Dict[str, Dict[str, float]],
    ) -> None:
        """
        Merge latest snapshot data into a persistent cache.

        new_data: {layer: {"current_peak": float, "global_peak": float}}
        Only provided fields update; missing fields keep prior values.
        """
        if not new_data:
            return

        for layer, entry in new_data.items():
            if not isinstance(entry, dict):
                continue

            cur = entry.get("current_peak")
            gbl = entry.get("global_peak")
            rec = cache.get(layer)

            if rec is None:
                if cur is not None or gbl is not None:
                    cache[layer] = {
                        "current": float(cur if cur is not None else 0.0),
                        "global": float(
                            gbl if gbl is not None else (cur if cur is not None else 0.0)
                        ),
                    }
                continue

            if cur is not None:
                rec["current"] = float(cur)
            if gbl is not None:
                rec["global"] = max(float(gbl), float(rec.get("global", 0.0)))

    def compute_cost_map(
            self,
            layers: Dict[str, float],
            activation_cache: Dict[str, Dict[str, float]],
            gradient_cache: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Compute the total memory cost per layer:

            param_memory
          + activation_global_peak (if any)
          + gradient_global_peak (if any)

        Missing activation/gradient simply add 0.
        """
        cost = {}
        for layer, param_mem in layers.items():
            act_peak = activation_cache.get(layer, {}).get("global", 0.0)
            grad_peak = gradient_cache.get(layer, {}).get("global", 0.0)
            cost[layer] = float(param_mem) + float(act_peak) + float(grad_peak)
        return cost


    def compute_dashboard_data(self) -> Dict[str, Any]:
        """
        Dashboard-specific data using total memory (params+activation+gradient).

        Does NOT affect CLI or Notebook rendering.
        """
        # Load core data (param memory + caches)
        layer_snapshot = self._compute_layer_snapshot()
        act_snapshot = self._compute_snapshot(is_activation=True)
        grad_snapshot = self._compute_snapshot(is_activation=False)

        self._merge_cache(self._activation_cache, act_snapshot)
        self._merge_cache(self._gradient_cache, grad_snapshot)

        layers = layer_snapshot["layer_memory"]
        activation_cache = self._activation_cache
        gradient_cache = self._gradient_cache

        # COST-BASED SORTING
        cost_map = self.compute_cost_map(layers, activation_cache, gradient_cache)
        sorted_cost_items = sorted(cost_map.items(), key=lambda kv: kv[1], reverse=True)

        top_cost_items = sorted_cost_items[: self._top_n]
        other_cost_items = sorted_cost_items[self._top_n:]

        total_cost_sum = sum(cost_map.values())
        other_total_cost = sum(v for _, v in other_cost_items)
        other_pct = (other_total_cost / total_cost_sum * 100.0) if total_cost_sum else 0.0

        # Build formatted rows for dashboard
        top_rows = []
        for layer, cost in top_cost_items:
            row = {
                "layer": layer,
                "total_cost": cost,
                "param_memory": layers.get(layer, 0.0),
                "activation_peak": activation_cache.get(layer, {}).get("global", 0.0),
                "gradient_peak": gradient_cache.get(layer, {}).get("global", 0.0),
                "pct": (cost / total_cost_sum * 100.0) if total_cost_sum else 0.0,
            }
            top_rows.append(row)

        return {
            "model_index": layer_snapshot["model_index"],
            "total_cost_sum": total_cost_sum,
            # top-N sorted by total memory
            "top_layers": top_rows,
            # aggregated "Other" layers
            "other": {
                "total_cost": other_total_cost,
                "pct": other_pct,
            },
            # optional: full list
            "all_layers": sorted_cost_items,
        }


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
