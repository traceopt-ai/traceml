from typing import Optional
from dataclasses import dataclass


@dataclass
class BaseTraceEvent:
    name: str  # "layer_forward", "optimizer_step", so on...
    step_: Optional[int]  # attached at flush
    timestamp: Optional[int]
