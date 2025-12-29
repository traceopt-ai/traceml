import time
from dataclasses import dataclass
from queue import Queue, Full
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader

from .base_trace_event import BaseTraceEvent
from .steptimer import timed_region

# Batch event queue
_batch_event_queue: Queue = Queue(maxsize=1024)

def get_batch_event_queue() -> Queue:
    return _batch_event_queue


@dataclass
class BatchEvent(BaseTraceEvent):
    """
    Represents a yielded batch from a DataLoader.
    """
    batch_size: Optional[int]


def infer_batch_size(batch: Any) -> Optional[int]:
    """
    Infer batch size from a batch object.

    Supports:
      - Tensor
      - Dict[str, Tensor]
      - Tuple/List[Tensor]
    """
    if torch.is_tensor(batch):
        return batch.shape[0]

    if isinstance(batch, dict):
        for v in batch.values():
            if torch.is_tensor(v):
                return v.shape[0]

    if isinstance(batch, (list, tuple)):
        for v in batch:
            if torch.is_tensor(v):
                return v.shape[0]

    return None


_ORIG_DATALOADER_ITER = DataLoader.__iter__


def _traceml_dataloader_iter(self):
    it = _ORIG_DATALOADER_ITER(self)

    while True:
        try:
            with timed_region(name="_traceml_internal:dataloader_next", use_gpu=False):
                batch = next(it)
        except StopIteration:
            break

        # try:
        #     bs = infer_batch_size(batch)
        #     evt = BatchEvent(
        #         name="batch_event",
        #         timestamp=time.time(),
        #         batch_size=bs,
        #     )
        #     try:
        #         _batch_event_queue.put_nowait(evt)
        #     except Full:
        #         pass
        # except Exception:
        #     pass

        yield batch


def patch_dataloader():
    """
    Patch torch.utils.data.DataLoader.__iter__ once.
    Safe to call multiple times.
    """
    if getattr(DataLoader, "_traceml_patched", False):
        return

    DataLoader.__iter__ = _traceml_dataloader_iter
    DataLoader._traceml_patched = True
