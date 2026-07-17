# Shared fixture

To keep the comparison comparable across runs and across people, pin the
workload to one agreed fixture rather than each person choosing their own model,
batch size, or topology.

**Agreed shared fixture:** ResNet-18, DDP, single machine.

**Canonical fixture script + one-line run command:** _pending_ (to be linked here
once the shared fixture is published).

Until then, `workload_core.py` in this directory provides the single-GPU
ResNet-18 / Imagenette baseline used to produce the reference numbers. When the
canonical DDP fixture lands, point every config at it so the compute is identical
across tools and across people.
