import uuid
import datetime
import os
_SESSION_ID = None


def generate_session_id():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"session_{ts}_{rand}"


def get_session_id():
    global _SESSION_ID
    session_name = os.environ["TRACEML_SESSION_NAME"]
    if session_name == "":
        _SESSION_ID = generate_session_id()
    else:
        _SESSION_ID = session_name
    return _SESSION_ID


def reset_session_id():
    """For future multi-run tracking."""
    global _SESSION_ID
    _SESSION_ID = generate_session_id()
    return _SESSION_ID
