"""
Stdout/stderr sampler for TraceML.

This sampler persists captured process output to a per-rank log file and, for
local rank 0, also mirrors lines into the sampler database for transport.
"""

from __future__ import annotations

from pathlib import Path

from traceml.runtime.stdout_stderr_capture import StreamCapture
from traceml.samplers.base_sampler import BaseSampler
from traceml.samplers.runtime_context import resolve_runtime_context
from traceml.samplers.utils import ensure_session_dir


class StdoutStderrSampler(BaseSampler):
    """
    Sampler that captures stdout/stderr lines and stores them incrementally.
    Runs on all ranks.
    """

    def __init__(
        self,
        max_cache_lines: int = 20_000,
        log_filename: str = "stdout_stderr.log",
    ) -> None:
        super().__init__(
            sampler_name="Stdout/Stderr",
            table_name="stdout_stderr",
        )

        self.max_cache_lines = max_cache_lines
        self._ctx = resolve_runtime_context()

        if self._ctx.local_rank != 0:
            self.sender = None

        logs_dir = ensure_session_dir(
            logs_dir=self._ctx.logs_dir,
            session_id=self._ctx.session_id,
            rank=self._ctx.local_rank,
        )
        self.log_path = Path(logs_dir) / log_filename
        self.log_path.write_text(
            "[TraceML] New run started\n\n",
            encoding="utf-8",
        )

    def sample(self) -> None:
        capture = StreamCapture._stdout_stderr_capture
        if capture is None:
            return

        text = capture.read_buffer()
        if not text:
            return

        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return

        if self._ctx.local_rank == 0:
            for line in lines:
                self._add_record({"line": line})

        with self.log_path.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
