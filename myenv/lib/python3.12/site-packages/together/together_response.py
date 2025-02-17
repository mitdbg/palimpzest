from __future__ import annotations

from typing import Any, Dict


class TogetherResponse:
    """
    API Response class. Stores headers and response data.
    """

    def __init__(self, data: Dict[str, Any], headers: Dict[str, Any]):
        self._headers = headers
        self.data = data

    @property
    def request_id(self) -> str | None:
        """
        Fetches request id from headers
        """
        if "cf-ray" in self._headers:
            return str(self._headers["cf-ray"])
        return None

    @property
    def requests_remaining(self) -> int | None:
        """
        Number of requests remaining at current rate limit
        """
        if "x-ratelimit-remaining" in self._headers:
            return int(self._headers["x-ratelimit-remaining"])
        return None

    @property
    def processed_by(self) -> str | None:
        """
        Processing host server name
        """
        if "x-hostname" in self._headers:
            return str(self._headers["x-hostname"])
        return None

    @property
    def response_ms(self) -> int | None:
        """
        Server request completion time
        """
        if "x-total-time" in self._headers:
            h = self._headers["x-total-time"]
            return None if h is None else round(float(h))
        return None
