from dataclasses import dataclass
from typing import Optional


@dataclass
class BaseTraceEvent:
    name: str  # "optimizer_step", "dataloader_fetch", and so on.
    step_: Optional[int]  # attached at flush
    timestamp: Optional[int]
