import queue
import socket
import struct
import threading
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import msgspec

from traceml.loggers.error_log import get_error_logger


@dataclass(frozen=True)
class TCPConfig:
    host: str = "127.0.0.1"
    port: int = 29765
    backlog: int = 16
    recv_buf: int = 65536


class TCPServer:
    """
    Non-blocking TCP server for TraceML DDP telemetry.

    Design:
      - One background thread handles accept + recv
      - Messages are length-prefixed MessagePack frames
      - poll() is non-blocking and safe to call from runtime thread
    """

    def __init__(self, cfg: TCPConfig):
        self.cfg = cfg
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[Dict]" = queue.Queue()
        self.logger = get_error_logger("TraceML-TCPServer")

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self._sock.bind((self.cfg.host, self.cfg.port))
        self._sock.listen(self.cfg.backlog)

        self._thread = threading.Thread(
            target=self._run,
            name="TraceML-TCPServer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    def poll(self) -> Iterator[Dict]:
        """
        Drain all currently available messages.

        Non-blocking. Safe to call inside runtime loop.
        """
        while True:
            try:
                yield self._queue.get_nowait()
            except queue.Empty:
                break

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                conn, _ = self._sock.accept()
                conn.settimeout(1.0)
                threading.Thread(
                    target=self._handle_client,
                    args=(conn,),
                    daemon=True,
                ).start()
            except OSError:
                break
            except Exception:
                continue

    def _drain_frames(
        self,
        buffer: bytes,
        expected: Optional[int],
    ) -> tuple[list[bytes], bytes, Optional[int]]:
        frames: list[bytes] = []

        while True:
            if expected is None:
                if len(buffer) < 4:
                    break
                expected = struct.unpack("!I", buffer[:4])[0]
                buffer = buffer[4:]

            if len(buffer) < expected:
                break

            frames.append(buffer[:expected])
            buffer = buffer[expected:]
            expected = None

        return frames, buffer, expected

    def _handle_client(self, conn: socket.socket) -> None:
        buffer = b""
        expected: Optional[int] = None
        decoder = msgspec.msgpack.Decoder()

        try:
            while not self._stop_event.is_set():
                try:
                    data = conn.recv(self.cfg.recv_buf)
                    if not data:
                        break  # peer closed
                except socket.timeout:
                    continue  # idle
                except OSError:
                    break  # socket error

                buffer += data
                frames, buffer, expected = self._drain_frames(buffer, expected)
                for payload in frames:
                    try:
                        msg = decoder.decode(payload)
                        self._queue.put_nowait(msg)
                    except queue.Full:
                        pass  # drop on overflow
                    except Exception:
                        continue  # corrupted frame
        finally:
            try:
                conn.close()
            except Exception:
                pass


# TCP Client (worker ranks)
class TCPClient:
    """
    Best-effort TCP client for TraceML telemetry.

    Notes:
      - No reconnect logic (intentional for MVP)
      - send() never raises
      - If rank 0 dies, training continues unaffected
    """

    def __init__(self, cfg: TCPConfig):
        self.cfg = cfg
        self._sock: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._connected = False
        self.logger = get_error_logger("TraceML-TCPClient")

    def send(self, payload: Dict) -> None:
        """
        Send a single message using msgspec.msgpack.
        """
        try:
            self._ensure_connected()
            # Encode dict to binary MessagePack
            data = msgspec.msgpack.encode(payload)
            header = struct.pack("!I", len(data))
            with self._lock:
                self._sock.sendall(header + data)
        except Exception:
            self._close()

    def send_batch(self, payloads: list) -> None:
        """
        Send multiple payloads in a single TCP write.

        All payloads are encoded together as a msgpack list and written
        with one ``sendall()`` call, replacing N individual ``send()`` calls
        with a single kernel syscall.

        The aggregator's ``RemoteDBStore.ingest()`` detects the list envelope
        and dispatches each item individually — fully backward compatible.

        Notes
        -----
        - An empty ``payloads`` list is a no-op (no socket write).
        - A failure to connect or send closes the socket; the next tick will
          attempt to reconnect (same behaviour as :meth:`send`).
        """
        if not payloads:
            return
        try:
            self._ensure_connected()
            # Encode the whole list as one msgpack frame
            data = msgspec.msgpack.encode(payloads)
            header = struct.pack("!I", len(data))
            with self._lock:
                self._sock.sendall(header + data)
        except Exception:
            self._close()

    def close(self) -> None:
        self._close()

    def _ensure_connected(self) -> None:
        if self._connected:
            return

        with self._lock:
            if self._connected:
                return

            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect((self.cfg.host, self.cfg.port))
            self._sock = sock
            self._connected = True

    def _close(self) -> None:
        with self._lock:
            if self._sock:
                try:
                    self._sock.close()
                except Exception:
                    pass
            self._sock = None
            self._connected = False
