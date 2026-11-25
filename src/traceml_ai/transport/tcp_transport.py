import queue
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Optional

from traceml_ai.loggers.error_log import get_error_logger
from traceml_ai.utils.msgpack_codec import Decoder as MsgpackDecoder
from traceml_ai.utils.msgpack_codec import encode as encode_msgpack


@dataclass(frozen=True)
class TCPConfig:
    host: str = "127.0.0.1"
    port: int = 29765
    backlog: int = 16
    recv_buf: int = 65536


def bind_exclusive_listener(
    host: str, port: int, backlog: int
) -> socket.socket:
    """
    Return a TCP socket listening on ``(host, port)`` that owns the port.

    Raises ``OSError`` (``EADDRINUSE``) while another socket listens there.
    The socket is closed before any error propagates.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Never SO_REUSEPORT: it would let a second aggregator (e.g. one
        # orphaned by a killed run) bind the same port, and the kernel would
        # then split rank connections between the two. POSIX SO_REUSEADDR
        # only permits rebinding over TIME_WAIT, which fast restarts need.
        # Windows SO_REUSEADDR also allows taking over a live port, so there
        # the socket asks for exclusive use instead.
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, int(port)))
        sock.listen(backlog)
    except BaseException:
        sock.close()
        raise
    return sock


class TCPServer:
    """
    Non-blocking TCP server for TraceML DDP telemetry.

    Design:
      - One background thread handles accept + recv
      - Messages are length-prefixed MessagePack frames
      - poll() is non-blocking and safe to call from runtime thread
      - ``first_frame_reply`` sees the first message of each connection. When
        it returns a dict, that dict goes back to the peer as one frame and
        the message is not queued. The server never sends otherwise.
    """

    def __init__(
        self,
        cfg: TCPConfig,
        first_frame_reply: Optional[Callable[[Any], Optional[Dict]]] = None,
    ):
        self.cfg = cfg
        self._first_frame_reply = first_frame_reply
        self._port = int(cfg.port)
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._queue: "queue.Queue[Dict]" = queue.Queue()
        self._data_ready = (
            threading.Event()
        )  # set by _handle_client on every new message
        # Accepted peer connections that have not closed yet. A rank that dies
        # closes its socket, so this is how the aggregator tells a rank that
        # may still send from one that never will.
        self._open_connections = 0
        self._connections_lock = threading.Lock()
        self.logger = get_error_logger("TraceML-TCPServer")

    @property
    def port(self) -> int:
        """Actual bound TCP port. Differs from config when cfg.port == 0."""
        return int(self._port)

    def start(self) -> None:
        self._sock = bind_exclusive_listener(
            self.cfg.host, self.cfg.port, self.cfg.backlog
        )
        self._port = int(self._sock.getsockname()[1])

        self._thread = threading.Thread(
            target=self._run,
            name="TraceML-TCPServer",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        sock = self._sock
        self._sock = None
        if sock:
            try:
                # On Linux close() alone does not wake the accept thread, and
                # its blocked accept() keeps the port listening after stop.
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                # Listening sockets commonly report ENOTCONN. shutdown() is
                # only a best-effort wakeup before the required close().
                pass
            try:
                sock.close()
            except Exception as exc:
                self.logger.error(
                    "[TraceML] TCP listener close failed: %s: %s",
                    type(exc).__name__,
                    exc,
                )

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

    def wait_for_data(self, timeout: float) -> bool:
        """
        Block until at least one message is available or ``timeout`` expires.

        Returns
        -------
        bool
            True if a message arrived before the timeout; False if the timeout
            expired with no data.  The caller should then call ``poll()`` to
            drain whatever is available.

        Notes
        -----
        - The internal event is cleared before this method returns so that the
          next call will block correctly (no spurious immediate wakeup).
        - This method is safe to call from any thread.
        """
        signalled = self._data_ready.wait(timeout=timeout)
        self._data_ready.clear()
        return signalled

    def open_connections(self) -> int:
        """
        Number of accepted peer connections that have not closed yet.

        A connection is counted from the moment its reader starts until the
        reader has queued every frame it received and closed the socket, so
        a count of zero means no queued frame is still on its way in.
        """
        with self._connections_lock:
            return self._open_connections

    def _track_connection(self, delta: int) -> None:
        with self._connections_lock:
            self._open_connections += delta

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
        buffer: bytearray,
        expected: Optional[int],
    ) -> tuple[list[bytes], bytearray, Optional[int]]:
        frames: list[bytes] = []
        offset = 0
        buf_len = len(buffer)

        while True:
            if expected is None:
                if buf_len - offset < 4:
                    break
                expected = struct.unpack("!I", buffer[offset : offset + 4])[0]
                offset += 4

            if buf_len - offset < expected:
                break

            frames.append(buffer[offset : offset + expected])
            offset += expected
            expected = None

        if offset > 0:
            del buffer[:offset]

        return frames, buffer, expected

    def _handle_client(self, conn: socket.socket) -> None:
        buffer = bytearray()  # mutable extend() is O(1) amortised, no copies
        expected: Optional[int] = None
        decoder = MsgpackDecoder()
        first_frame = True

        self._track_connection(1)
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

                buffer.extend(data)
                frames, buffer, expected = self._drain_frames(buffer, expected)
                for payload in frames:
                    try:
                        msg = decoder.decode(payload)
                        if first_frame:
                            first_frame = False
                            if self._reply_to_first_frame(conn, msg):
                                continue
                        self._queue.put_nowait(msg)
                        self._data_ready.set()  # wake the aggregator loop
                    except queue.Full:
                        pass  # drop on overflow
                    except Exception:
                        continue  # corrupted frame
        finally:
            try:
                conn.close()
            except Exception:
                pass
            # Last, so every frame from this peer is queued before the count
            # can reach zero.
            self._track_connection(-1)

    def _reply_to_first_frame(self, conn: socket.socket, msg: Any) -> bool:
        """Send the reply for a connection's first message, if it has one.

        Returns True when the message was answered, so it is not queued.
        """
        if self._first_frame_reply is None:
            return False
        try:
            reply = self._first_frame_reply(msg)
        except Exception as exc:
            self.logger.error(
                "[TraceML] TCP first-frame reply failed: %s: %s",
                type(exc).__name__,
                exc,
            )
            return False
        if reply is None:
            return False
        try:
            conn.sendall(_encode_frame(reply))
        except OSError:
            pass  # the peer left; nothing to queue either way
        return True


def _encode_frame(payload: Any) -> bytes:
    data = encode_msgpack(payload)
    return struct.pack("!I", len(data)) + data


class NoReplyError(Exception):
    """The peer sent nothing back: it hung up or stayed silent."""


def request_one_frame(
    sock: socket.socket,
    payload: Any,
    *,
    timeout_sec: float,
    max_frame_bytes: int = 65536,
) -> Any:
    """Send one frame on ``sock``, half-close it, and decode one reply frame.

    The half-close tells the peer nothing more is coming, so a peer that
    does not answer hangs up at once instead of letting ``timeout_sec``
    run out.

    Raises ``NoReplyError`` when the peer closes or stays silent for
    ``timeout_sec`` without sending a byte. Raises ``ValueError`` when it
    sends anything but one msgpack frame of at most ``max_frame_bytes``,
    including a frame cut short.
    """
    deadline = time.monotonic() + float(timeout_sec)
    sock.settimeout(float(timeout_sec))
    sock.sendall(_encode_frame(payload))
    sock.shutdown(socket.SHUT_WR)
    buffer = bytearray()
    length: Optional[int] = None
    while length is None or len(buffer) < 4 + length:
        remaining = deadline - time.monotonic()
        data = b""
        if remaining > 0:
            sock.settimeout(remaining)
            try:
                data = sock.recv(4096)
            except socket.timeout:
                pass
        if not data:
            if not buffer:
                raise NoReplyError()
            raise ValueError("reply cut short")
        buffer.extend(data)
        if length is None and len(buffer) >= 4:
            length = struct.unpack("!I", bytes(buffer[:4]))[0]
            if length > max_frame_bytes:
                raise ValueError("reply is not a TraceML frame")
    try:
        return MsgpackDecoder().decode(bytes(buffer[4 : 4 + length]))
    except Exception as exc:
        raise ValueError("reply is not msgpack") from exc


# TCP Client (worker ranks)
class TCPClient:
    """
    Best-effort TCP client for TraceML telemetry.

    Notes:
      - The next send after a failure reconnects (no background loop)
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
            data = encode_msgpack(payload)
            header = struct.pack("!I", len(data))
            with self._lock:
                self._sock.sendall(header + data)
        except Exception as exc:
            self._handle_send_failure("send", exc)

    def send_batch(self, payloads: list) -> None:
        """
        Send multiple payloads in a single TCP write.

        All payloads are encoded together as a msgpack list and written
        with one ``sendall()`` call, replacing N individual ``send()`` calls
        with a single kernel syscall.

        The aggregator detects the list envelope, dispatches each item
        individually, and writes the resulting rows into SQLite-backed
        history.

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
            data = encode_msgpack(payloads)
            header = struct.pack("!I", len(data))
            with self._lock:
                self._sock.sendall(header + data)
        except Exception as exc:
            self._handle_send_failure("send_batch", exc)

    def close(self) -> None:
        self._close()

    def _handle_send_failure(self, operation: str, exc: Exception) -> None:
        """Close the broken connection and record why telemetry was lost."""
        self._close()
        self.logger.error(
            "[TraceML] TCP telemetry %s failed for %s:%s: %s: %s",
            operation,
            self.cfg.host,
            self.cfg.port,
            type(exc).__name__,
            exc,
        )

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
