# Distributed Training

TraceML uses launch flags similar to `torchrun` and writes the same
`final_summary.json` artifact for single-node and multi-node summary runs.
`--html-report` additionally writes `final_summary.html`; in multi-node runs it
is produced only on the aggregator-owner node (node 0), like the JSON/TXT
artifacts.

## Single-node DDP/FSDP

Use `--nproc-per-node` for a single machine with multiple local workers:

```bash
traceml run train.py --nproc-per-node=4
```

Live CLI and dashboard modes are intended for single-node runs. Summary mode is
the default and works for both single-node and multi-node summary reports.

## Multi-node DDP

Use the same `--run-name`, `--nnodes`, `--nproc-per-node`, and
`--master-addr` on every node. Set `--node-rank` to the current node's rank.

On node 0:

```bash
traceml run train.py \
  --nnodes=2 \
  --node-rank=0 \
  --nproc-per-node=4 \
  --master-addr=<node0-ip> \
  --run-name=my-run
```

On node 1:

```bash
traceml run train.py \
  --nnodes=2 \
  --node-rank=1 \
  --nproc-per-node=4 \
  --master-addr=<node0-ip> \
  --run-name=my-run
```

Node 0 starts the TraceML aggregator. Other nodes connect to
`<node0-ip>:29765` by default. If workers need a different reachable address or
port for TraceML telemetry, add `--aggregator-host=<host>` or
`--aggregator-port=<port>` on every node.

For multi-node runs, node 0 binds the aggregator to `0.0.0.0` by default.
Override that only when needed with `--aggregator-bind-host=<bind-host>`.

`--session-id` remains accepted as a backward-compatible alias for
`--run-name`.

At the end of a summary run, node 0 waits for rank-finished markers, drains
late telemetry, checkpoints SQLite, and then writes `final_summary.*`. The
default finalization budget is 300 seconds. On slow shared filesystems or
congested networks, increase it with `--finalize-timeout-sec <seconds>` on each
node, or set `TRACEML_FINALIZE_TIMEOUT_SEC`.

## Running on Slurm

On a Slurm-managed cluster, derive these flags from the job environment instead
of setting them by hand. See [Running on Slurm](slurm.md) for the mapping and a
copy-paste sbatch template.
