from abc import ABC, abstractmethod
from traceml.database.database import Database


class BaseSampler(ABC):
    """
    Abstract base class for samplers that monitor runtime metrics,
    such as CPU usage, tensor allocations, or custom events.

    Samplers may be stateful and are typically polled periodically.
    """

    def __init__(self, sampler_name) -> None:
        self.db = Database(sampler_name=sampler_name)

    @abstractmethod
    def sample(self):
        raise NotImplementedError("Must be implemented by subclasses.")
