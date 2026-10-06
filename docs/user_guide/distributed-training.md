# Distributed Training

Run distributed training through TraceML to get per-rank timing and a final
bottleneck diagnosis. TraceML launches workers through `torchrun`; your script
must already be configured for distributed training.

Standard Hugging Face Trainer, Lightning, and RF-DETR training paths attach
TraceML automatically. Custom PyTorch loops need explicit setup. Check your
[integration guide](integrations.md) for the strategies it supports.

## Single-node DDP/FSDP

For four local GPU workers:

```bash
traceml run train.py --nproc-per-node=4
```

Use the worker count your training configuration expects. For Lightning,
match `devices` to the local worker count and `num_nodes` to the node count. The launcher does
not convert a single-device script into DDP or FSDP. FSDP timing support depends
on the integration; it is not automatically enabled for every trainer.

Summary mode is the default. For a single-node live view, add `--mode=cli` or
`--mode=dashboard`. Dashboard mode requires
`pip install "traceml-ai[dashboard]"`.

## Multi-node DDP

Start one TraceML launcher per node. Each launcher starts its local workers.
The following example uses two nodes with four workers each.

Before launching:

- Use the same training code and environment on both nodes.
- Choose a fresh run name and a `--logs-dir` on storage visible to both nodes.
  TraceML refuses to overwrite an existing run folder.
- Replace `node0` with node 0's reachable hostname or IP, and
  `/shared/traceml/logs` with your shared log directory.
- Allow connections to node 0 on ports `29500` (torchrun) and `29765`
  (TraceML telemetry), or configure different ports below.

On node 0:

```bash
traceml run train.py \
  --nnodes=2 \
  --node-rank=0 \
  --nproc-per-node=4 \
  --master-addr=node0 \
  --run-name=my-run \
  --logs-dir=/shared/traceml/logs
```

On node 1:

```bash
traceml run train.py \
  --nnodes=2 \
  --node-rank=1 \
  --nproc-per-node=4 \
  --master-addr=node0 \
  --run-name=my-run \
  --logs-dir=/shared/traceml/logs
```

Keep the launch settings identical except for `--node-rank`. Node 0 creates
the shared run folder and starts the TraceML aggregator; the other nodes join
that run. Multi-node runs use summary mode.

## Read the result

After telemetry finalizes successfully, node 0 writes the final diagnosis to:

```text
<logs-dir>/<run-name>/final_summary.json
<logs-dir>/<run-name>/final_summary.txt
```

Add `--html-report` to generate `final_summary.html` in the same folder.
The final reports are written by node 0 only.

Each node saves its training stdout and stderr under
`nodes/node_<node-rank>/`. Node 0 also saves aggregator stderr under
`aggregator/process.stderr.log` and reports final telemetry health.

See [Reading the Output](reading-output.md) for rank timing and diagnosis,
[Compare Runs](compare.md) to compare saved reports, or
[Training Crashes](training-crashes.md) if training fails.

## Running on Slurm

Use Slurm to start one launcher per node and derive node ranks from the job
environment. See [Running on Slurm](slurm.md) for the existing job template.

## Advanced launch settings

### Network addresses and ports

| Setting | Default | When to change it |
| --- | --- | --- |
| `--master-port` | `29500` | Use another port for torchrun rendezvous. |
| `--aggregator-port` | `29765` | Use another port for TraceML telemetry. |
| `--aggregator-host` | The master address on multi-node runs | Workers reach node 0 through a different network address. |
| `--aggregator-bind-host` | `0.0.0.0` on multi-node runs | Bind the aggregator to a specific interface on node 0. |

Pass the same network settings on every node. If an earlier aggregator still
owns the telemetry endpoint, the default policy stops the new launch. Stop
the earlier process or choose another aggregator port. See
[Missing-aggregator behavior](public-api.md#missing-aggregator-behavior)
for the optional continue-without-telemetry policy.

### Finalization timeout

At shutdown, node 0 waits for rank completion and drains telemetry before
writing the reports. The default finalization budget is 300 seconds. On slow
shared storage or congested networks, increase it with
`--finalize-timeout-sec <seconds>` on every node, or set
`TRACEML_FINALIZE_TIMEOUT_SEC`.

### Older launch commands

`--session-id` remains an alias for `--run-name`. Both identify the same shared
run and require a fresh name for each launch.
