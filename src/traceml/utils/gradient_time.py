import sys
import time
from dataclasses import dataclass, asdict
from queue import Queue, Full
from typing import Optional, Dict, Any

gradient_time_queue: Queue = Queue(maxsize=2048)


@dataclass
class GradientTimeEvent:
    """
    Represents timing of one backward/optimizer step.
    """

    model_id: int
    timestamp: float
    label: str
    backward_time: float
    optimizer_time: float
    total_time: float
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StepTimer:
    """
    Context manager + markers to measure gradient/backward/optimizer timings.

    Usage:
        with StepTimer(model, label="train") as t:
            t.mark_backward_start()
            loss.backward()
            t.mark_backward_done()

            t.mark_optimizer_step_start()
            optimizer.step()
            t.mark_optimizer_step_done()
    """

    def __init__(self, model, label: str = "train"):
        self.model_id = id(model)
        self.label = label
        self._backward_start: Optional[float] = None
        self._backward_done: Optional[float] = None
        self._optim_start: Optional[float] = None
        self._optim_done: Optional[float] = None
        self._context_start: Optional[float] = None

    def __enter__(self):
        self._context_start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            now = time.time()
            total = now - (self._context_start or now)

            backward_time = (
                (self._backward_done - self._backward_start)
                if self._backward_start and self._backward_done
                else 0.0
            )
            optimizer_time = (
                (self._optim_done - self._optim_start)
                if self._optim_start and self._optim_done
                else 0.0
            )

            event = GradientTimeEvent(
                model_id=self.model_id,
                timestamp=now,
                label=self.label,
                backward_time=round(backward_time, 6),
                optimizer_time=round(optimizer_time, 6),
                total_time=round(total, 6),
            )
            try:
                gradient_time_queue.put_nowait(event)
            except Full:
                pass
        except Exception as e:
            print(f"[TraceML] StepTimer error: {e}", file=sys.stderr)

    def mark_backward_start(self):
        self._backward_start = time.time()

    def mark_backward_done(self):
        self._backward_done = time.time()

    def mark_optimizer_step_start(self):
        self._optim_start = time.time()

    def mark_optimizer_step_done(self):
        self._optim_done = time.time()
