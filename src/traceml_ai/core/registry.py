"""
Small typed registry helper.

Registries are useful for extension points where TraceML wants stable names for
components such as samplers, diagnostic rules, display drivers, or summary
sections without hardcoding lookups throughout core orchestration code.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """
    Insertion-ordered mapping from stable string keys to typed objects.

    The registry intentionally keeps a small surface area:
    - duplicate registration fails immediately
    - missing lookup fails with a clear error
    - iteration and ``all()`` preserve registration order
    """

    def __init__(
        self,
        items: Optional[Iterable[tuple[str, T]]] = None,
    ) -> None:
        self._items: "OrderedDict[str, T]" = OrderedDict()
        if items is not None:
            for key, item in items:
                self.register(key, item)

    def register(self, key: str, item: T) -> T:
        """
        Register an item under a stable key and return the item.

        Returning the item makes decorator-style registration possible later
        without adding another API shape.
        """
        normalized = self._normalize_key(key)
        if normalized in self._items:
            raise KeyError(f"Registry key already registered: {normalized!r}")
        self._items[normalized] = item
        return item

    def get(self, key: str) -> T:
        """Return the registered item for ``key``."""
        normalized = self._normalize_key(key)
        try:
            return self._items[normalized]
        except KeyError as exc:
            raise KeyError(
                f"Registry key is not registered: {normalized!r}"
            ) from exc

    def keys(self) -> tuple[str, ...]:
        """Return registered keys in registration order."""
        return tuple(self._items.keys())

    def all(self) -> tuple[T, ...]:
        """Return all registered items in registration order."""
        return tuple(self._items.values())

    def items(self) -> tuple[tuple[str, T], ...]:
        """Return registered key/item pairs in registration order."""
        return tuple((key, item) for key, item in self._items.items())

    def as_mapping(self) -> Mapping[str, T]:
        """Return a shallow read-only mapping view of registered items."""
        return self._items.copy()

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return self._normalize_key(key) in self._items

    def __iter__(self) -> Iterator[T]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    @staticmethod
    def _normalize_key(key: str) -> str:
        normalized = str(key).strip()
        if not normalized:
            raise ValueError("Registry key cannot be empty.")
        return normalized


__all__ = [
    "Registry",
]
