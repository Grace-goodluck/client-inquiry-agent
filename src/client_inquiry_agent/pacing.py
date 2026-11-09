"""One pacer shared by every outgoing call: Gemini and the Second Brain alike.

Each brain /ask makes its own Gemini call on the same free-tier quota, so all outgoing calls,
retries included, start at least MIN_CALL_SPACING_S apart. One message makes one Gemini call,
so in practice this spaces retries and back-to-back requests. Thread-safe, so the web app
can share one instance across requests.
"""

import threading
import time
from collections.abc import Callable

from client_inquiry_agent.config import MIN_CALL_SPACING_S


class Pacer:
    """Start-to-start spacing between outgoing calls."""

    def __init__(
        self,
        min_interval: float = MIN_CALL_SPACING_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last_start: float | None = None
        self._lock = threading.Lock()

    def wait_turn(self) -> None:
        with self._lock:
            if self._last_start is not None:
                remaining = self._last_start + self.min_interval - self._clock()
                if remaining > 0:
                    self._sleep(remaining)
            self._last_start = self._clock()
