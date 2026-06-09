"""
DDP gradient-sync timing demo.

Shows TraceML timing each DDP gradient all_reduce during
``loss.backward()`` via the comm hook, emitted per bucket as
``_traceml_comm:ddp_grad_sync`` events.

Run with torchrun (2 ranks shown; per-step timings printed from the
in-process queue at the end — each rank prints its own table)::

    torchrun --nnodes=2 --nproc_per_node=1 --node_rank=<0|1> \\
        --master_addr=<node0-ip> --master_port=29400 \\
        examples/ddp_comm_hook_demo.py

Or via the TraceML CLI (full pipeline; per-step comm timings are read back
from the session SQLite DB and printed at the end)::

    traceml run examples/ddp_comm_hook_demo.py --mode summary \\
        --run-name ddp_comm_demo --nnodes 2 --node-rank <0|1> \\
        --master-addr <node0-ip> --aggregator-bind-host 0.0.0.0

Convention
----------
- ``traceml.init(mode="auto")`` patches ``DDP.forward`` so the comm hook
  auto-installs on each DDP model's first forward (no explicit call needed).
  ``traceml.wrap_ddp(model)`` is the explicit equivalent and is required in
  ``manual`` / ``selective`` mode.
- Pass the DDP **wrapper** (not ``model.module``) to ``trace_step(model)``
  so step-time and layer-buffer flushes route by the right model id.
- Pass ``model.module`` (the inner module) to ``trace_model_instance()``
  for layer hooks.

Notes
-----
- During ``ddp_model.no_sync()`` accumulation steps, DDP suppresses the
  hook — those steps produce zero ``ddp_grad_sync`` events (correct:
  no communication happened).
- Comm timing overlaps with backward compute for early buckets. Only the
  last bucket's all_reduce is pure communication (backward is done).
  See ``deep_dive_bucket_ordering_and_overlap.md`` for details.
- Data-only for now: ``_traceml_comm:ddp_grad_sync`` events are persisted to
  the telemetry DB but are NOT yet surfaced in the terminal/dashboard or the
  diagnosis engine (until then the comm wall-time stays folded into the
  ``wait_ms`` residual, as ``wait_ms`` always has for DDP). Wiring a comm
  bucket + comm-bound diagnosis is the TRA-30 follow-up.
- The session DB lives on the node-0 machine (the aggregator owner), so the
  DB-backed timing table prints on global rank 0 only.
"""

import glob
import json
import os
import socket
import sqlite3
import sys
from queue import Empty

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

import traceml_ai as traceml

SEED = 42
INPUT_DIM = 512
HIDDEN_DIM = 1024
NUM_CLASSES = 10
BATCH_SIZE = 512
NUM_SAMPLES = 8192
NUM_STEPS = 30

COMM_EVENT = "_traceml_comm:ddp_grad_sync"
DATA_EVENT = "_traceml_internal:dataloader_next"
H2D_EVENT = "_traceml_internal:h2d_time"
FWD_EVENT = "_traceml_internal:forward_time"
BWD_EVENT = "_traceml_internal:backward_time"
OPT_EVENT = "_traceml_internal:optimizer_step"
STEP_EVENT = "_traceml_internal:step_time"

# (column label, event name) — every step-scoped stream TraceML records,
# in training-step execution order. Comm is the TRA-16 addition.
COLUMNS = [
    ("Data", DATA_EVENT),
    ("H2D", H2D_EVENT),
    ("Fwd", FWD_EVENT),
    ("Bwd", BWD_EVENT),
    ("Opt", OPT_EVENT),
    ("Comm", COMM_EVENT),
    ("Total", STEP_EVENT),
]


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


def _print_header(lead: str) -> int:
    """Print the timing-table header; return its width for separators."""
    header = (
        lead + "".join(f"{label:>9}" for label, _ in COLUMNS) + f"{'Bkts':>6}"
    )
    print("\n" + header + "   (all times in ms)", flush=True)
    print("-" * len(header), flush=True)
    return len(header)


def print_inprocess_results(use_cuda: bool) -> None:
    """
    Per-step timing table from the in-process step-time queue.

    Under plain torchrun there is no TraceML runtime/sampler, so the
    StepTimeBatch objects flushed by ``trace_step`` stay in the process-local
    queue. Draining it here is demo-only introspection — under
    ``traceml run`` the sampler consumes this queue and the data lands in
    the session SQLite DB instead (see ``print_db_results``).
    """
    from traceml_ai.utils.timing import get_step_time_queue

    if use_cuda:
        torch.cuda.synchronize()

    batches = []
    q = get_step_time_queue()
    while True:
        try:
            batches.append(q.get_nowait())
        except Empty:
            break

    if not batches:
        print(
            "No buffered step batches (running under `traceml run`? "
            "The sampler consumed them — see the DB table instead).",
            flush=True,
        )
        return

    def duration_ms(evt) -> float:
        evt.try_resolve()
        if evt.gpu_time_ms is not None:
            return evt.gpu_time_ms
        return (evt.cpu_end - evt.cpu_start) * 1000.0

    width = _print_header(f"{'Step':>5}")
    col_totals = {label: 0.0 for label, _ in COLUMNS}
    total_buckets = 0
    n = len(batches)

    for batch in sorted(batches, key=lambda b: b.step):
        # Sum per event name: a step can carry several events of the same
        # stream (e.g. one h2d per .to() call, one comm per DDP bucket).
        sums: dict = {}
        counts: dict = {}
        for evt in batch.events:
            d = duration_ms(evt)
            sums[evt.name] = sums.get(evt.name, 0.0) + d
            counts[evt.name] = counts.get(evt.name, 0) + 1

        buckets = counts.get(COMM_EVENT, 0)
        total_buckets += buckets

        row = f"{batch.step:>5}"
        for label, name in COLUMNS:
            v = sums.get(name)
            row += f"{v:>9.2f}" if v is not None else f"{'-':>9}"
            if v is not None:
                col_totals[label] += v
        row += f"{buckets:>6}"
        print(row, flush=True)

    print("-" * width, flush=True)
    avg_row = f"{'avg':>5}"
    for label, _ in COLUMNS:
        avg_row += f"{col_totals[label] / n:>9.2f}"
    avg_row += f"{total_buckets / n:>6.1f}"
    print(avg_row, flush=True)


def print_db_results(db_path: str) -> None:
    """Per-step, per-rank timing table read back from the session DB."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]

    if "step_time_samples" not in tables:
        print(
            "step_time_samples table not found — aggregator may not have "
            "flushed.",
            flush=True,
        )
        conn.close()
        return

    cur.execute(
        "SELECT step, rank, events_json FROM step_time_samples "
        "ORDER BY step, rank"
    )
    rows = cur.fetchall()
    conn.close()

    _print_header(f"{'Step':>5} {'Rank':>5}")

    for step, rank, events_json in rows:
        events = json.loads(events_json) if events_json else {}

        def stat(event_name, key="duration_ms", fmt="{:.2f}"):
            entry = events.get(event_name, {})
            for device_stats in entry.values():
                v = device_stats.get(key)
                if v is not None:
                    return fmt.format(v)
            return "-"

        row = f"{step:>5} {rank:>5}"
        for _, name in COLUMNS:
            row += f"{stat(name):>9}"
        row += f"{stat(COMM_EVENT, key='n_calls', fmt='{:.0f}'):>6}"
        print(row, flush=True)


def main() -> None:
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    def log(msg: str) -> None:
        print(f"[rank {rank}] {msg}", flush=True)

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        device_name = torch.cuda.get_device_name(local_rank)
    else:
        device = torch.device("cpu")
        device_name = "cpu"

    traceml.init(mode="auto")

    torch.manual_seed(SEED)
    inner_model = TinyMLP().to(device)

    n_params = sum(p.numel() for p in inner_model.parameters())
    grad_mb = n_params * 4 / (1024 * 1024)
    batches_per_epoch = NUM_SAMPLES // world_size // BATCH_SIZE

    if rank == 0:
        print("=" * 70, flush=True)
        print("DDP gradient-sync timing demo (TRA-16)", flush=True)
        print("=" * 70, flush=True)
        print(f"world_size       : {world_size} ranks", flush=True)
        print(f"backend          : {backend}", flush=True)
        print(
            f"model            : TinyMLP {INPUT_DIM}->{HIDDEN_DIM}"
            f"->{NUM_CLASSES} ({n_params:,} params, "
            f"{grad_mb:.2f} MB fp32 grads to all_reduce per step)",
            flush=True,
        )
        print(
            f"data             : {NUM_SAMPLES} samples, batch {BATCH_SIZE} "
            f"-> {batches_per_epoch} batches/epoch/rank",
            flush=True,
        )
        print(
            f"steps            : {NUM_STEPS} "
            f"(~{-(-NUM_STEPS // max(batches_per_epoch, 1))} epochs)",
            flush=True,
        )
        print(
            "what is timed    : every DDP gradient all_reduce (per bucket) "
            f"during loss.backward(), emitted as {COMM_EVENT}",
            flush=True,
        )
        print("=" * 70, flush=True)

    log(f"host={socket.gethostname()} device={device_name}")

    if use_cuda:
        model = torch.nn.parallel.DistributedDataParallel(
            inner_model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    else:
        model = torch.nn.parallel.DistributedDataParallel(inner_model)

    # Explicit wrap_ddp. Optional in auto mode: traceml.init(mode="auto")
    # above already auto-installs the comm hook on the first forward.
    # It is required in manual / selective mode.
    traceml.wrap_ddp(model)
    log(
        "comm hook installed="
        f"{getattr(model, '_traceml_ddp_comm_hook_installed', False)} "
        "(explicit wrap_ddp; init(mode='auto') would also auto-install "
        "on first forward)"
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    x = torch.randn(NUM_SAMPLES, INPUT_DIM)
    y = torch.randint(0, NUM_CLASSES, (NUM_SAMPLES,))
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        pin_memory=use_cuda,
        drop_last=True,
    )

    model.train()
    step = 0
    epoch = 0

    # Epoch loop so NUM_STEPS is honored even when it exceeds one epoch.
    while step < NUM_STEPS:
        sampler.set_epoch(epoch)
        for batch_x, batch_y in loader:
            if step >= NUM_STEPS:
                break

            with traceml.trace_step(model):
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

            step += 1
            if rank == 0 and step % 5 == 0:
                print(
                    f"step {step:>3}/{NUM_STEPS} | epoch {epoch} | "
                    f"loss {loss.item():.4f}",
                    flush=True,
                )
        epoch += 1

    # Sanity: after DDP all_reduce, every rank must hold identical grads.
    # (The comm hook only TIMES the sync; it must not alter gradients.)
    grad_norm = torch.sqrt(
        sum((p.grad.detach() ** 2).sum() for p in inner_model.parameters())
    )
    norms = [torch.zeros_like(grad_norm) for _ in range(world_size)]
    dist.all_gather(norms, grad_norm)
    if rank == 0:
        norm_strs = ", ".join(
            f"r{i}={n.item():.6f}" for i, n in enumerate(norms)
        )
        identical = all(
            torch.allclose(norms[0], n, rtol=1e-6, atol=1e-9) for n in norms
        )
        print(
            f"\ngrad-norm check  : {norm_strs} | identical across ranks: "
            f"{identical}",
            flush=True,
        )

    under_launcher = bool(os.environ.get("TRACEML_SESSION_ID", "").strip())

    if not under_launcher:
        # Plain torchrun: no sampler drained the step-time queue, so each
        # rank can print its own per-step table from in-process data.
        log("per-step timings (in-process, this rank only):")
        print_inprocess_results(use_cuda)
    else:
        # Under `traceml run`: the sampler shipped everything to the
        # aggregator; read the per-rank timings back from the session DB.
        traceml.final_summary()
        if rank == 0:
            db_path = find_db()
            if db_path:
                print(f"\n[demo] session DB: {db_path}", flush=True)
                print_db_results(db_path)
            else:
                print(
                    "DB not found on this node (it lives on node 0).",
                    file=sys.stderr,
                )

    if rank == 0:
        print(
            "\nWhat you just saw: TraceML's comm hook wrapped DDP's "
            "gradient all_reduce and timed every bucket with CUDA events "
            "while training ran at full speed. Early buckets overlap "
            "backward compute; only the last bucket is pure comm wait. "
            f"Under `traceml run` these land in SQLite as {COMM_EVENT} "
            "(dashboard/diagnosis wiring is the TRA-30 follow-up).",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
