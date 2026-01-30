import datetime
import uuid


def _generate_session_id():
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rand = uuid.uuid4().hex[:6]
    return f"session_{ts}_{rand}"


def get_session_id():
    _SESSION_ID = _generate_session_id()
    return _SESSION_ID
