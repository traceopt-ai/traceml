"""
H2D Transfer Timing Demo
========================
Demonstrates that TraceML records host-to-device (.to(device)) transfer
time as a first-class signal alongside forward, backward, and optimizer timing.

Each training step moves a batch from CPU RAM to GPU via:
    batch_x = batch_x.to(device, non_blocking=True)
    batch_y = batch_y.to(device, non_blocking=True)

Because traceml.init(mode="auto") patches torch.Tensor.to(), those calls are
timed and stored in the SQLite DB under the event name:
    _traceml_internal:h2d_time

After training, this script queries the DB directly and prints the H2D
timings per step so you can see the data with no extra tooling.

Usage
-----
    traceml run examples/h2d_timing_demo.py

Requires a CUDA GPU for meaningful H2D timings (works on CPU too, timings
will just be near-zero).
"""

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
    """
    Locate the TraceML SQLite DB for this session.

    The aggregator writes to ``{session_root}/aggregator/telemetry`` (no
    .db extension). The launcher sets TRACEML_SESSION_ID and TRACEML_LOGS_DIR
    in the executor environment, so we can derive the path directly.
    """
    session_id = os.environ.get("TRACEML_SESSION_ID", "").strip()
    logs_dir = os.environ.get("TRACEML_LOGS_DIR", "").strip()
    if session_id and logs_dir:
        # logs_dir may be relative — resolve from cwd set by the launcher
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

    # Fallback: scan common log roots for any 'telemetry' file
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
            return matches[-1]  # most recent session

    return None


def print_h2d_results(db_path: str) -> None:
    """Query the DB and print H2D timings per step."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print(f"\nTables in DB: {tables}")

    if "step_time_samples" not in tables:
        print(
            "step_time_samples table not found — aggregator may not have flushed."
        )
        conn.close()
        return

    cur.execute(
        "SELECT step, events_json FROM step_time_samples ORDER BY step"
    )
    rows = cur.fetchall()
    conn.close()

    print(
        f"\n{'Step':>5}  {'H2D (ms)':>10}  {'Forward (ms)':>13}  {'Backward (ms)':>14}"
    )
    print("-" * 50)

    for step, events_json in rows:
        events = json.loads(events_json) if events_json else {}

        def ms(event_name):
            entry = events.get(event_name, {})
            for device_stats in entry.values():
                d = device_stats.get("duration_ms")
                if d is not None:
                    return f"{d:.2f}"
            return "—"

        print(
            f"{step:>5}  "
            f"{ms('_traceml_internal:h2d_time'):>10}  "
            f"{ms('_traceml_internal:forward_time'):>13}  "
            f"{ms('_traceml_internal:backward_time'):>14}"
        )


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

    for batch_x, batch_y in loader:
        if step >= NUM_STEPS:
            break

        with traceml.trace_step(model):
            # These .to() calls happen inside trace_step so they are timed
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        step += 1
        if step % 10 == 0:
            print(f"Step {step}/{NUM_STEPS} | loss: {loss.item():.4f}")

    print("\nTraining done. Waiting for final summary...")
    traceml.final_summary(print_text=True)

    # Print the DB path to stderr so it appears in the terminal AFTER the TUI
    # exits (TUI captures stdout; stderr goes to the panel but the path line
    # will also be echoed to the terminal by the launcher).
    db_path = find_db()
    if db_path:
        print(f"\n[H2D demo] DB path: {db_path}", file=sys.stderr)
        print(
            "[H2D demo] Query H2D timings with:\n"
            f"  sqlite3 {db_path} "
            '"SELECT step, events_json FROM step_time_samples ORDER BY step;"',
            file=sys.stderr,
        )
        print_h2d_results(db_path)
    else:
        print(
            "DB not found. Make sure you ran with: traceml run examples/h2d_timing_demo.py",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
