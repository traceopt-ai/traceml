import sys
import threading
from io import StringIO


class StreamCapture(StringIO):
    """
    Simple thread-safe capture utility for stdout/stderr.
    """

    _stdout_stderr_capture = None  ## shared instance
    _orig_stdout = sys.__stdout__
    _orig_stderr = sys.__stderr__
    _redirected = False
    _lock = threading.Lock()

    def __init__(self):
        super().__init__()
        self._local_lock = threading.Lock()

    def write(self, data: str):
        with self._local_lock:
            super().write(data)
            super().flush()
        return len(data)

    def read_buffer(self) -> str:
        with self._local_lock:
            return self.getvalue()

    @classmethod
    def redirect_to_capture(cls):
        with cls._lock:
            if not cls._redirected:
                cls._stdout_stderr_capture = StreamCapture()
                sys.stdout = cls._stdout_stderr_capture
                sys.stderr = cls._stdout_stderr_capture
                cls._redirected = True

    @classmethod
    def redirect_to_original(cls):
        with cls._lock:
            if cls._redirected:
                sys.stdout = cls._orig_stdout
                sys.stderr = cls._orig_stderr
                cls._redirected = False
