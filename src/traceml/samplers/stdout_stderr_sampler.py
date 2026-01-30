from pathlib import Path

from traceml.config import config
from traceml.distributed import get_ddp_info
from traceml.samplers.base_sampler import BaseSampler
from traceml.stdout_stderr_capture import StreamCapture


class StdoutStderrSampler(BaseSampler):
    """
    Sampler that captures stdout/stderr lines and stores them incrementally.
    Runs on all ranks.
    """

    sampler_name = "Stdout/Stderr"

    def __init__(
        self,
        max_cache_lines: int = 20_000,
        log_filename: str = "stdout_stderr.log",
    ):
        super().__init__(sampler_name=self.sampler_name)
        self.enable_ddp_send = False  # per-rank logs only
        self.max_cache_lines = max_cache_lines
        # per-rank log file (unchanged semantics)
        session_id = config.session_id
        _, local_rank, _ = get_ddp_info()

        logs_dir = Path(config.logs_dir) / session_id / str(local_rank)
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = logs_dir / log_filename
        self.log_path.write_text(
            "[TraceML] New run started\n\n",
            encoding="utf-8",
        )

    def sample(self):
        capture = StreamCapture._stdout_stderr_capture
        if capture is None:
            return
        text = capture.read_buffer() if capture else ""
        if not text:
            return
        lines = [ln for ln in text.splitlines() if ln.strip()]
        if not lines:
            return
        # append to DB
        for ln in lines:
            self.db.add_record(
                "stdout_stderr",
                {
                    "line": ln,
                },
            )

        # persist to file
        with self.log_path.open("a", encoding="utf-8") as f:
            for ln in lines:
                f.write(ln + "\n")
