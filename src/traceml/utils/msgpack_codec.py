"""
Small MessagePack-style codec facade used by TraceML internals.

Why this exists
---------------
TraceML prefers ``msgspec`` for framed binary payloads, but some environments
used for lightweight tooling or tests may not have a working installation
available. This module provides one narrow internal abstraction so callers can
encode/decode framed payloads consistently without scattering backend-specific
imports and fallbacks across the codebase.

Backend selection
-----------------
1. Use ``msgspec`` when it is importable and passes a simple round-trip probe.
2. Fall back to UTF-8 JSON bytes otherwise.

The JSON fallback preserves correctness for TraceML's own producer/consumer
paths because both ends use the same helper. The on-disk and on-wire framing
remains unchanged: a 4-byte big-endian length prefix followed by opaque bytes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class _CodecBackend:
    encode: Callable[[Any], bytes]
    decode: Callable[[bytes], Any]
    name: str


def _json_backend() -> _CodecBackend:
    def _encode(payload: Any) -> bytes:
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def _decode(payload: bytes) -> Any:
        return json.loads(payload.decode("utf-8"))

    return _CodecBackend(encode=_encode, decode=_decode, name="json")


def _msgspec_backend() -> _CodecBackend | None:
    try:
        import msgspec
    except ModuleNotFoundError:
        return None

    try:
        probe = {"_traceml_probe": 1}
        encoded = msgspec.msgpack.encode(probe)
        decoded = msgspec.msgpack.Decoder().decode(encoded)
    except Exception:
        return None

    if not isinstance(encoded, (bytes, bytearray)) or not encoded:
        return None
    if decoded != probe:
        return None

    return _CodecBackend(
        encode=msgspec.msgpack.encode,
        decode=msgspec.msgpack.Decoder().decode,
        name="msgspec",
    )


_BACKEND = _msgspec_backend() or _json_backend()


def encode(payload: Any) -> bytes:
    """Encode one payload into framed-body bytes."""
    return _BACKEND.encode(payload)


def decode(payload: bytes) -> Any:
    """Decode one framed-body payload."""
    return _BACKEND.decode(payload)


class Encoder:
    """Tiny state-free encoder wrapper matching the narrow usage in TraceML."""

    def encode(self, payload: Any) -> bytes:
        return encode(payload)


class Decoder:
    """Tiny state-free decoder wrapper matching the narrow usage in TraceML."""

    def decode(self, payload: bytes) -> Any:
        return decode(payload)


def backend_name() -> str:
    """Return the active backend name for debugging or diagnostics."""
    return _BACKEND.name
