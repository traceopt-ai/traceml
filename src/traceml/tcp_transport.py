import pickle
import struct
import queue
import socket
import threading
from dataclasses import dataclass
from typing import Dict, Iterator, Optional
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
      - Messages are newline-delimited JSON (NDJSON)
      - poll() is non-blocking and safe to call from runtime thread
    """

    def __init__(self, cfg: TCPConfig):
        self.cfg = cfg
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[Dict]" = queue.Queue()

    def start(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
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
                        msg = pickle.loads(payload)
                    except Exception:
                        continue  # corrupted frame

                    try:
                        self._queue.put_nowait(msg)
                    except queue.Full:
                        pass  # drop on overflow
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
        Send a single JSON message (newline-delimited).

        Best-effort: silently drops on failure.
        """
        try:
            self._ensure_connected()
            data = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
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
