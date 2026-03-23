"""
Suggest GPU Display Driver
"""

import os
import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from traceml.aggregator.display_drivers.base import BaseDisplayDriver
from traceml.aggregator.hardware_catalog import recommend_hardware
from traceml.database.remote_database_store import RemoteDBStore
from traceml.runtime.settings import TraceMLSettings


class SuggestDisplayDriver(BaseDisplayDriver):
    def __init__(
        self, logger: Any, store: RemoteDBStore, settings: TraceMLSettings
    ) -> None:
        self._logger = logger
        self._store = store
        self._settings = settings
        self._console = Console()
        self._target_batch_size = int(
            os.environ.get("TRACEML_TARGET_BATCH_SIZE", "1")
        )

    def start(self) -> None:
        self._console.print(
            "[bold cyan]TraceML Suggest-GPU[/bold cyan] profiling in progress (running ~3 steps)..."
        )

    def tick(self) -> None:
        pass

    def stop(self) -> None:
        # Give aggregator a moment to ingest final rows
        time.sleep(1.0)

        # Pull data
        layer_db = self._store.get_db(
            rank=0, sampler_name="LayerMemorySampler"
        )
        step_db = self._store.get_db(rank=0, sampler_name="StepMemorySampler")

        if not layer_db or not step_db:
            self._console.print(
                "[bold red]Failed to collect enough telemetry (need more steps/model). Make sure you run at least 3 steps.[/bold red]"
            )
            return

        last_layer = layer_db.get_last_record("LayerMemoryTable")
        last_step = step_db.get_last_record("step_memory")

        if not last_layer or not last_step:
            self._console.print(
                "[bold red]No records found in sampler tables. Did the script run any steps?[/bold red]"
            )
            return

        # Get parameter memory
        param_bytes = last_layer.get("total_param_bytes", 0)

        # Get peak memory (already in bytes, not MB!)
        peak_allocated = last_step.get("peak_alloc", 0)

        # Math
        param_gb = param_bytes / (1024**3)
        grad_gb = param_gb
        optimizer_gb = param_gb * 2  # Assume Adam

        base_mem_gb = param_gb + grad_gb + optimizer_gb

        peak_gb = peak_allocated / (1024**3)
        # Activations are whatever is left after param/grad/opt
        activations_gb = max(0, peak_gb - base_mem_gb)

        # Extrapolate activations
        extrapolated_activations = activations_gb * self._target_batch_size

        total_required_gb = base_mem_gb + extrapolated_activations

        # Recommend hardware
        recommendation = recommend_hardware(total_required_gb)

        # Render Table
        table = Table(
            title=f"Hardware Recommendation (Target Extrapolation: x{self._target_batch_size})"
        )
        table.add_column("Component", style="cyan")
        table.add_column("Estimated VRAM (GB)", justify="right", style="green")

        table.add_row("Model Parameters", f"{param_gb:.2f}")
        table.add_row("Gradients", f"{grad_gb:.2f}")
        table.add_row("Optimizer States (Adam)", f"{optimizer_gb:.2f}")
        table.add_row(
            f"Activations (Extrapolated x{self._target_batch_size})",
            f"{extrapolated_activations:.2f}",
        )
        table.add_row(
            "Total Estimated VRAM",
            f"{total_required_gb:.2f}",
            style="bold yellow",
        )

        self._console.print()
        self._console.print(
            Panel(
                table, title="[bold magenta]TraceML Suggest GPU[/bold magenta]"
            )
        )
        self._console.print(
            f"\n[bold green]Recommended Hardware:[/bold green] {recommendation}"
        )
        self._console.print(
            "[dim]Note: This assumes Adam optimizer and runs local script as baseline (usually bs=1).[/dim]\n"
        )
