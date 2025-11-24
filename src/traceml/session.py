import uuid
import datetime

_SESSION_ID = None


def generate_session_id():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"session_{ts}_{rand}"


def get_session_id():
    global _SESSION_ID
    if _SESSION_ID is None:
        _SESSION_ID = generate_session_id()
    return _SESSION_ID


def reset_session_id():
    """For future multi-run tracking."""
    global _SESSION_ID
    _SESSION_ID = generate_session_id()
    return _SESSION_ID
