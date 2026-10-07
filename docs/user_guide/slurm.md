# Running on Slurm

Submit a distributed training job through TraceML using the existing Slurm
template. Slurm starts one TraceML launcher per node; each launcher starts
its local workers through `torchrun`.

Your script must already support distributed training and use a supported
automatic trainer or [explicit integration](integrations.md).

## 1. Prepare the job

The template contains two files in
[`examples/distributed/slurm/`](https://github.com/traceopt-ai/traceml/tree/main/examples/distributed/slurm):

- `traceml_ddp.sbatch`: requests resources and starts one task per node.
- `launch.sh`: expands each node's Slurm variables and launches training.

Before submitting:

- Adjust the node count, GPUs, CPUs, time limit, and any account or partition
  settings in the sbatch file for your cluster.
- Activate your training environment before `srun`. TraceML and your framework
  must be available on every node.
- Keep `--ntasks-per-node=1`. This template uses torchrun to create the GPU
  workers; Slurm must not create another task for every GPU.
- Put the scripts and log directory on shared storage. The template uses
  `./logs` under the submit directory; change `--logs-dir` in `launch.sh` if
  needed.

The supplied wrapper runs `examples/distributed/ddp_minimal.py`, which already
contains explicit TraceML setup. Replace that path with your training script
and match its distributed configuration to the requested resources.

## 2. Submit

From the TraceML repository root:

```bash
sbatch examples/distributed/slurm/traceml_ddp.sbatch
```

The job uses two nodes with four GPUs each by default. Node 0 is the first host
in the allocation and owns both torchrun rendezvous and the TraceML aggregator.
Summary mode is the default for multi-node runs.

## 3. Read the result

After telemetry finalizes successfully, the report is saved under:

```text
logs/ddp-<job-id>-<restart-count>/final_summary.json
logs/ddp-<job-id>-<restart-count>/final_summary.txt
```

Each node's training streams are saved under `nodes/node_<node-rank>/` in that
run folder. Slurm also saves the batch output as `traceml-<job-id>.out`.

Add `--html-report` to the wrapper for an HTML report. See
[Reading the Output](reading-output.md), [Compare Runs](compare.md), or
[Training Crashes](training-crashes.md) for the next step.

## How the template launches training

The sbatch file exports settings shared by every node:

```bash
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export RUN_NAME="ddp-${SLURM_JOB_ID}-${SLURM_RESTART_COUNT:-0}"
srun examples/distributed/slurm/launch.sh
```

The wrapper expands the per-node settings where it runs:

```bash
exec traceml run examples/distributed/ddp_minimal.py \
  --mode=summary \
  --run-name="${RUN_NAME}" \
  --nnodes="${SLURM_NNODES}" \
  --node-rank="${SLURM_NODEID}" \
  --nproc-per-node="${SLURM_GPUS_ON_NODE}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port=29500
```

| TraceML setting | Slurm source |
| --- | --- |
| `--nnodes` | `SLURM_NNODES` |
| `--node-rank` | `SLURM_NODEID`, expanded on each node |
| `--nproc-per-node` | `SLURM_GPUS_ON_NODE`, the allocated GPU count |
| `--master-addr` | First host in `SLURM_JOB_NODELIST` |
| `--run-name` | Job ID plus restart count, shared by all nodes |

Use a fresh run name for each launch. The restart count distinguishes requeued
jobs, which retain their job ID. TraceML refuses to overwrite an existing run
folder.

### Expand the node rank on each node

Keep the launch command in the wrapper. This inline command is incorrect:

```bash
# The batch shell expands the rank once, before srun launches the other nodes.
srun traceml run train.py --node-rank=$SLURM_NODEID ...
```

It can give every node rank 0. Running `launch.sh` once per node lets each
process read its own `SLURM_NODEID`.

## Cluster adjustments

### GPU allocation and environment

The template uses `--gres=gpu:4`; some clusters require
`--gpus-per-node=4`. Ensure each launcher sees all GPUs allocated to its node
and that `SLURM_GPUS_ON_NODE` provides the intended numeric worker count.
`SLURM_GPUS_PER_NODE` may include a GPU type and is not used by the wrapper.

For CPU training, set an explicit local worker count in the wrapper instead
of deriving it from GPUs. Keep one launcher task per node.

If your site resets task environments, also activate the environment inside
`launch.sh`. Keep slow setup before the launch so node 0 can start the
aggregator promptly.

### Network access

Other nodes must reach node 0 on two ports:

| Purpose | Default port |
| --- | ---: |
| torchrun rendezvous | `29500` |
| TraceML telemetry | `29765` |

If the first host's name is not reachable, set `MASTER_ADDR` to a reachable
node-0 IP. For separate telemetry interfaces, custom ports, or bind addresses,
use the [advanced distributed settings](distributed-training.md#advanced-launch-settings).

For NCCL connection problems, your cluster may require
`NCCL_SOCKET_IFNAME=<interface>`. `NCCL_DEBUG=INFO` helps inspect that setup.

### Slow finalization

The finalization budget is 300 seconds by default. For slow shared storage or
late telemetry, add `--finalize-timeout-sec <seconds>` to the wrapper or set
`TRACEML_FINALIZE_TIMEOUT_SEC`. This controls telemetry shutdown, not Slurm's
job time limit.
