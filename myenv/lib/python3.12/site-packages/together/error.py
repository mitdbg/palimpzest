from __future__ import annotations

import json
from typing import Any, Dict

from requests import RequestException

from together.types.error import TogetherErrorResponse


class TogetherException(Exception):
    def __init__(
        self,
        message: (
            TogetherErrorResponse | Exception | str | RequestException | None
        ) = None,
        headers: str | Dict[Any, Any] | None = None,
        request_id: str | None = None,
        http_status: int | None = None,
    ) -> None:
        _message = (
            json.dumps(message.model_dump(exclude_none=True))
            if isinstance(message, TogetherErrorResponse)
            else message
        )
        if http_status is not None:
            self._message = f"Error code: {http_status} - {_message}"
        else:
            self._message = str(_message)

        super().__init__(self._message)

        self.http_status = http_status
        self.headers = headers or {}
        self.request_id = request_id

    def __repr__(self) -> str:
        repr_message = json.dumps(
            {
                "response": self._message,
                "status": self.http_status,
                "request_id": self.request_id,
                "headers": self.headers,
            }
        )
        return "%s(%r)" % (self.__class__.__name__, repr_message)


class AuthenticationError(TogetherException):
    def __init__(
        self,
        message: (
            TogetherErrorResponse | Exception | str | RequestException | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)


class ResponseError(TogetherException):
    def __init__(
        self,
        message: (
            TogetherErrorResponse | Exception | str | RequestException | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)


class JSONError(TogetherException):
    def __init__(
        self,
        message: (
            TogetherErrorResponse | Exception | str | RequestException | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)


class InstanceError(TogetherException):
    def __init__(self, model: str | None = "model", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.message = f"""No running instances for {model}.
                You can start an instance with one of the following methods:
                  1. navigating to the Together Playground at api.together.ai
                  2. starting one in python using together.Models.start(model_name)
                  3. `$ together models start <MODEL_NAME>` at the command line.
                See `together.Models.list()` in python or `$ together models list` in command line
                to get an updated list of valid model names.
                """


class RateLimitError(TogetherException):
    def __init__(
        self,
        message: (
            TogetherErrorResponse | Exception | str | RequestException | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)


class FileTypeError(TogetherException):
    def __init__(
        self,
        message: (
            TogetherErrorResponse | Exception | str | RequestException | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)


class AttributeError(TogetherException):
    def __init__(
        self,
        message: (
            TogetherErrorResponse | Exception | str | RequestException | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)


class Timeout(TogetherException):
    def __init__(
        self,
        message: (
            TogetherErrorResponse | Exception | str | RequestException | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)


class APIConnectionError(TogetherException):
    def __init__(
        self,
        message: (
            TogetherErrorResponse | Exception | str | RequestException | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)


class InvalidRequestError(TogetherException):
    def __init__(
        self,
        message: (
            TogetherErrorResponse | Exception | str | RequestException | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)


class APIError(TogetherException):
    def __init__(
        self,
        message: (
            TogetherErrorResponse | Exception | str | RequestException | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)


class ServiceUnavailableError(TogetherException):
    def __init__(
        self,
        message: (
            TogetherErrorResponse | Exception | str | RequestException | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)


class DownloadError(TogetherException):
    def __init__(
        self,
        message: (
            TogetherErrorResponse | Exception | str | RequestException | None
        ) = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(message=message, **kwargs)
