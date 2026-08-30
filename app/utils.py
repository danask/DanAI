"""
Shared utility helpers used across services.
"""
import time


def current_timestamp_ms() -> int:
    """Returns the current time as a Unix epoch timestamp in milliseconds (UTC).

    Using a single numeric format everywhere avoids the mismatch that came
    from mixing timezone-naive local-time ISO strings (chat/word-memory
    timestamps) with timezone-aware UTC datetime objects (news timestamps),
    which forced the frontend to guess at timezones when rendering dates.
    """
    return int(time.time() * 1000)
