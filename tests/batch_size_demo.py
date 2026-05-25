import glob
import json
import os
import sqlite3
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import traceml

SEED = 42
INPUT_DIM = 512
HIDDEN_DIM = 1024
NUM_CLASSES = 10
BATCH_SIZE = 512
NUM_SAMPLES = 8192
NUM_STEPS = 30
# Gradient accumulation: more than one micro-batch per step exercises the
# sum-across-step semantics in BatchSizeSampler.
GRAD_ACCUM_STEPS = 2


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


def find_db() -> str | None:
    """Locate the TraceML SQLite DB for this session."""
    session_id = os.environ.get("TRACEML_SESSION_ID", "").strip()
    logs_dir = os.environ.get("TRACEML_LOGS_DIR", "").strip()
    if session_id and logs_dir:
        launch_cwd = (
            os.environ.get("TRACEML_LAUNCH_CWD", "").strip() or os.getcwd()
        )
        logs_dir_abs = (
            os.path.join(launch_cwd, logs_dir)
            if not os.path.isabs(logs_dir)
            else logs_dir
        )
        db = os.path.join(logs_dir_abs, session_id, "aggregator", "telemetry")
        if os.path.exists(db):
            return db

    for root in [
        os.path.join(os.path.dirname(__file__), "logs"),
        os.path.join(os.getcwd(), "logs"),
    ]:
        matches = sorted(
            glob.glob(
                os.path.join(root, "**/aggregator/telemetry"), recursive=True
            )
        )
        if matches:
            return matches[-1]

    return None


def _expected_bytes_per_step() -> int:
    """
    Expected per-step H2D bytes given the demo's batch shape and dtypes.

    Each step inside trace_step does:
        batch_x = batch_x.to(device, non_blocking=True)  # float32
        batch_y = batch_y.to(device, non_blocking=True)  # int64
    repeated GRAD_ACCUM_STEPS times.
    """
    bx = BATCH_SIZE * INPUT_DIM * 4  # float32
    by = BATCH_SIZE * 8  # int64
    return GRAD_ACCUM_STEPS * (bx + by)


def print_batch_size_results(db_path: str) -> None:
    """Query the DB and print batch_size + h2d timing per step, per rank."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {r[0] for r in cur.fetchall()}
    print(f"\nTables in DB: {sorted(tables)}")

    if "batch_size_samples" not in tables:
        print(
            "batch_size_samples table not found — aggregator may not have flushed."
        )
        conn.close()
        return

    cur.execute(
        """
        SELECT global_rank, step, bytes_total, n_transfers, sample_ts_s
        FROM batch_size_samples
        ORDER BY global_rank, step;
        """
    )
    rows = cur.fetchall()

    print(
        f"\n{'Rank':>5}  {'Step':>5}  {'Bytes':>14}  "
        f"{'n_transfers':>11}  {'H2D (ms)':>10}"
    )
    print("-" * 60)

    # Pull H2D timing alongside for sanity-checking
    cur.execute(
        """
        SELECT global_rank, step, events_json
        FROM step_time_samples;
        """
    )
    timing_index = {}
    for global_rank, step, events_json in cur.fetchall():
        events = json.loads(events_json) if events_json else {}
        h2d_entry = events.get("_traceml_internal:h2d_time", {})
        ms_val = None
        for device_stats in h2d_entry.values():
            d = device_stats.get("duration_ms")
            if d is not None:
                ms_val = d
                break
        timing_index[(global_rank, step)] = ms_val

    for global_rank, step, bytes_total, n_transfers, _ts in rows:
        ms_val = timing_index.get((global_rank, step))
        ms_str = f"{ms_val:.2f}" if ms_val is not None else "—"
        print(
            f"{global_rank if global_rank is not None else '—':>5}  "
            f"{step:>5}  {bytes_total:>14,}  {n_transfers:>11}  {ms_str:>10}"
        )

    expected = _expected_bytes_per_step()
    print(f"\nExpected bytes/step (per rank): {expected:,}")
    conn.close()


def main():
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    traceml.init(mode="auto")

    # Keep data on CPU so each step does a real H2D transfer
    x = torch.randn(NUM_SAMPLES, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    dataset = TensorDataset(x, y)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = TinyMLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    step = 0
    loader_iter = iter(loader)

    while step < NUM_STEPS:
        with traceml.trace_step(model):
            optimizer.zero_grad(set_to_none=True)

            # Multiple micro-batches per step exercise SUM aggregation.
            for _ in range(GRAD_ACCUM_STEPS):
                try:
                    batch_x, batch_y = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    batch_x, batch_y = next(loader_iter)

                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                logits = model(batch_x)
                loss = criterion(logits, batch_y) / GRAD_ACCUM_STEPS
                loss.backward()

            optimizer.step()

        step += 1
        if step % 10 == 0:
            print(f"Step {step}/{NUM_STEPS} | loss: {loss.item():.4f}")

    print("\nTraining done. Waiting for final summary...")
    traceml.final_summary()

    db_path = find_db()
    if db_path:
        print(f"\n[BatchSize demo] DB path: {db_path}", file=sys.stderr)
        print(
            "[BatchSize demo] Install sqlite3 (Ubuntu/Debian RunPod):\n"
            "  apt-get update && apt-get install -y sqlite3\n\n"
            "[BatchSize demo] Inspect rows per (rank, step):\n"
            f"  sqlite3 -header -column {db_path} \\\n"
            '    "SELECT global_rank, step, bytes_total, n_transfers '
            'FROM batch_size_samples ORDER BY global_rank, step;"\n\n'
            "[BatchSize demo] Per-rank totals (sanity check):\n"
            f"  sqlite3 -header -column {db_path} \\\n"
            '    "SELECT global_rank, COUNT(*) AS steps, '
            "SUM(bytes_total) AS total_bytes "
            'FROM batch_size_samples GROUP BY global_rank;"\n\n'
            "[BatchSize demo] Show table schema:\n"
            f'  sqlite3 {db_path} ".schema batch_size_samples"',
            file=sys.stderr,
        )
        print_batch_size_results(db_path)
    else:
        print(
            "DB not found. Make sure you ran with: "
            "traceml run tests/batch_size_demo.py",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
