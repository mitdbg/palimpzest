from __future__ import annotations

import asyncio
import email.utils
import json
import sys
import threading
import time
from json import JSONDecodeError
from random import random
from typing import (
    Any,
    AsyncContextManager,
    AsyncGenerator,
    Dict,
    Iterator,
    Tuple,
    overload,
)
from urllib.parse import urlencode, urlsplit, urlunsplit

import aiohttp
import requests
from tqdm.utils import CallbackIOWrapper


if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import together
from together import error, utils
from together.constants import (
    BASE_URL,
    INITIAL_RETRY_DELAY,
    MAX_CONNECTION_RETRIES,
    MAX_RETRIES,
    MAX_RETRY_DELAY,
    MAX_SESSION_LIFETIME_SECS,
    TIMEOUT_SECS,
)
from together.together_response import TogetherResponse
from together.types import TogetherClient, TogetherRequest
from together.types.error import TogetherErrorResponse


# Has one attribute per thread, 'session'.
_thread_context = threading.local()


def _build_api_url(url: str, query: str) -> str:
    scheme, netloc, path, base_query, fragment = urlsplit(url)

    if base_query:
        query = "%s&%s" % (base_query, query)

    return str(urlunsplit((scheme, netloc, path, query, fragment)))


def _make_session(max_retries: int | None = None) -> requests.Session:
    if together.requestssession:
        if isinstance(together.requestssession, requests.Session):
            return together.requestssession
        return together.requestssession()
    s = requests.Session()
    s.mount(
        "https://",
        requests.adapters.HTTPAdapter(max_retries=max_retries),
    )
    return s


def parse_stream_helper(line: bytes) -> str | None:
    if line and line.startswith(b"data:"):
        if line.startswith(b"data: "):
            # SSE event may be valid when it contains whitespace
            line = line[len(b"data: ") :]
        else:
            line = line[len(b"data:") :]
        if line.strip() == b"[DONE]":
            # return here will cause GeneratorExit exception in urllib3
            # and it will close http connection with TCP Reset
            return None
        else:
            return line.decode("utf-8")
    return None


def parse_stream(rbody: Iterator[bytes]) -> Iterator[str]:
    for line in rbody:
        _line = parse_stream_helper(line)
        if _line is not None:
            yield _line


async def parse_stream_async(rbody: aiohttp.StreamReader) -> AsyncGenerator[str, Any]:
    async for line in rbody:
        _line = parse_stream_helper(line)
        if _line is not None:
            yield _line


class APIRequestor:
    def __init__(self, client: TogetherClient):
        self.api_base = client.base_url or BASE_URL
        self.api_key = client.api_key or utils.default_api_key()
        self.retries = MAX_RETRIES if client.max_retries is None else client.max_retries
        self.supplied_headers = client.supplied_headers
        self.timeout = client.timeout or TIMEOUT_SECS

    def _parse_retry_after_header(
        self, response_headers: Dict[str, Any] | None = None
    ) -> float | None:
        """
        Returns a float of the number of seconds (not milliseconds)
        to wait after retrying, or None if unspecified.

        About the Retry-After header:
            https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After
        See also
            https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After#syntax
        """
        if not response_headers:
            return None

        # First, try the non-standard `retry-after-ms` header for milliseconds,
        # which is more precise than integer-seconds `retry-after`
        try:
            retry_ms_header = response_headers.get("retry-after-ms", None)
            return float(retry_ms_header) / 1000
        except (TypeError, ValueError):
            pass

        # Next, try parsing `retry-after` header as seconds (allowing nonstandard floats).
        retry_header = str(response_headers.get("retry-after"))
        try:
            # note: the spec indicates that this should only ever be an integer
            # but if someone sends a float there's no reason for us to not respect it
            return float(retry_header)
        except (TypeError, ValueError):
            pass

        # Last, try parsing `retry-after` as a date.
        retry_date_tuple = email.utils.parsedate_tz(retry_header)
        if retry_date_tuple is None:
            return None

        retry_date = email.utils.mktime_tz(retry_date_tuple)
        return float(retry_date - time.time())

    def _calculate_retry_timeout(
        self,
        remaining_retries: int,
        response_headers: Dict[str, Any] | None = None,
    ) -> float:
        # If the API asks us to wait a certain amount of time (and it's a reasonable amount), just do what it says.
        retry_after = self._parse_retry_after_header(response_headers)
        if retry_after is not None and 0 < retry_after <= 60:
            return retry_after

        nb_retries = self.retries - remaining_retries

        # Apply exponential backoff, but not more than the max.
        sleep_seconds = min(INITIAL_RETRY_DELAY * pow(2.0, nb_retries), MAX_RETRY_DELAY)

        # Apply some jitter, plus-or-minus half a second.
        jitter = 1 - 0.25 * random()
        timeout = sleep_seconds * jitter
        return timeout if timeout >= 0 else 0

    def _retry_request(
        self,
        options: TogetherRequest,
        remaining_retries: int,
        response_headers: Dict[str, Any] | None,
        *,
        stream: bool,
        request_timeout: float | Tuple[float, float] | None = None,
    ) -> requests.Response:
        remaining = remaining_retries - 1
        if remaining == 1:
            utils.log_debug("1 retry left")
        else:
            utils.log_debug(f"{remaining} retries left")

        timeout = self._calculate_retry_timeout(remaining, response_headers)
        ("Retrying request to %s in %f seconds", options.url, timeout)

        # In a synchronous context we are blocking the entire thread. Up to the library user to run the client in a
        # different thread if necessary.
        time.sleep(timeout)

        return self.request_raw(
            options=options,
            stream=stream,
            request_timeout=request_timeout,
            remaining_retries=remaining,
        )

    @overload
    def request(
        self,
        options: TogetherRequest,
        stream: Literal[True],
        remaining_retries: int | None = ...,
        request_timeout: float | Tuple[float, float] | None = ...,
    ) -> Tuple[Iterator[TogetherResponse], bool, str]:
        pass

    @overload
    def request(
        self,
        options: TogetherRequest,
        stream: Literal[False] = ...,
        remaining_retries: int | None = ...,
        request_timeout: float | Tuple[float, float] | None = ...,
    ) -> Tuple[TogetherResponse, bool, str]:
        pass

    @overload
    def request(
        self,
        options: TogetherRequest,
        stream: bool = ...,
        remaining_retries: int | None = ...,
        request_timeout: float | Tuple[float, float] | None = ...,
    ) -> Tuple[TogetherResponse | Iterator[TogetherResponse], bool, str]:
        pass

    def request(
        self,
        options: TogetherRequest,
        stream: bool = False,
        remaining_retries: int | None = None,
        request_timeout: float | Tuple[float, float] | None = None,
    ) -> Tuple[
        TogetherResponse | Iterator[TogetherResponse],
        bool,
        str | None,
    ]:
        result = self.request_raw(
            options=options,
            remaining_retries=remaining_retries or self.retries,
            stream=stream,
            request_timeout=request_timeout,
        )

        resp, got_stream = self._interpret_response(result, stream)
        return resp, got_stream, self.api_key

    @overload
    async def arequest(
        self,
        options: TogetherRequest,
        stream: Literal[True],
        request_timeout: float | Tuple[float, float] | None = ...,
    ) -> Tuple[AsyncGenerator[TogetherResponse, None], bool, str]:
        pass

    @overload
    async def arequest(
        self,
        options: TogetherRequest,
        *,
        stream: Literal[True],
        request_timeout: float | Tuple[float, float] | None = ...,
    ) -> Tuple[AsyncGenerator[TogetherResponse, None], bool, str]:
        pass

    @overload
    async def arequest(
        self,
        options: TogetherRequest,
        stream: Literal[False] = ...,
        request_timeout: float | Tuple[float, float] | None = ...,
    ) -> Tuple[TogetherResponse, bool, str]:
        pass

    @overload
    async def arequest(
        self,
        options: TogetherRequest,
        stream: bool = ...,
        request_timeout: float | Tuple[float, float] | None = ...,
    ) -> Tuple[TogetherResponse | AsyncGenerator[TogetherResponse, None], bool, str]:
        pass

    async def arequest(
        self,
        options: TogetherRequest,
        stream: bool = False,
        request_timeout: float | Tuple[float, float] | None = None,
    ) -> Tuple[TogetherResponse | AsyncGenerator[TogetherResponse, None], bool, str]:
        ctx = AioHTTPSession()
        session = await ctx.__aenter__()
        result = None
        try:
            result = await self.arequest_raw(
                options,
                session,
                request_timeout=request_timeout,
            )
            resp, got_stream = await self._interpret_async_response(result, stream)
        except Exception:
            # Close the request before exiting session context.
            if result is not None:
                result.release()
            await ctx.__aexit__(None, None, None)
            raise
        if got_stream:

            async def wrap_resp() -> AsyncGenerator[TogetherResponse, None]:
                assert isinstance(resp, AsyncGenerator)
                try:
                    async for r in resp:
                        yield r
                finally:
                    # Close the request before exiting session context. Important to do it here
                    # as if stream is not fully exhausted, we need to close the request nevertheless.
                    result.release()
                    await ctx.__aexit__(None, None, None)

            return wrap_resp(), got_stream, self.api_key  # type: ignore
        else:
            # Close the request before exiting session context.
            result.release()
            await ctx.__aexit__(None, None, None)
            return resp, got_stream, self.api_key  # type: ignore

    @classmethod
    def handle_error_response(
        cls,
        resp: TogetherResponse,
        rcode: int,
        stream_error: bool = False,
    ) -> Exception:
        try:
            assert isinstance(resp.data, dict)
            error_resp = resp.data.get("error")
            assert isinstance(
                error_resp, dict
            ), f"Unexpected error response {error_resp}"
            error_data = TogetherErrorResponse(**(error_resp))
        except (KeyError, TypeError):
            raise error.JSONError(
                "Invalid response object from API: %r (HTTP response code "
                "was %d)" % (resp.data, rcode),
                http_status=rcode,
            )

        utils.log_info(
            "Together API error received",
            error_code=error_data.code,
            error_type=error_data.type_,
            error_message=error_data.message,
            error_param=error_data.param,
            stream_error=stream_error,
        )

        # Rate limits were previously coded as 400's with code 'rate_limit'
        if rcode == 429:
            return error.RateLimitError(
                error_data,
                http_status=rcode,
                headers=resp._headers,
                request_id=resp.request_id,
            )
        elif rcode in [400, 403, 404, 415]:
            return error.InvalidRequestError(
                error_data,
                http_status=rcode,
                headers=resp._headers,
                request_id=resp.request_id,
            )
        elif rcode == 401:
            return error.AuthenticationError(
                error_data,
                http_status=rcode,
                headers=resp._headers,
                request_id=resp.request_id,
            )

        elif stream_error:
            parts = [error_data.message, "(Error occurred while streaming.)"]
            message = " ".join([p for p in parts if p is not None])
            return error.APIError(
                message,
                http_status=rcode,
                headers=resp._headers,
                request_id=resp.request_id,
            )
        else:
            return error.APIError(
                error_data,
                http_status=rcode,
                headers=resp._headers,
                request_id=resp.request_id,
            )

    @classmethod
    def _validate_headers(
        cls, supplied_headers: Dict[str, str] | None
    ) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if supplied_headers is None:
            return headers

        if not isinstance(supplied_headers, dict):
            raise TypeError("Headers must be a dictionary")

        for k, v in supplied_headers.items():
            if not isinstance(k, str):
                raise TypeError("Header keys must be strings")
            if not isinstance(v, str):
                raise TypeError("Header values must be strings")
            headers[k] = v

        # NOTE: It is possible to do more validation of the headers, but a request could always
        # be made to the API manually with invalid headers, so we need to handle them server side.

        return headers

    def _prepare_request_raw(
        self,
        options: TogetherRequest,
        absolute: bool = False,
    ) -> Tuple[str, Dict[str, str], Dict[str, str] | CallbackIOWrapper | bytes | None]:
        abs_url = options.url if absolute else "%s%s" % (self.api_base, options.url)
        headers = self._validate_headers(options.headers or self.supplied_headers)

        data = None
        data_bytes = None
        if options.method.lower() == "get" or options.method.lower() == "delete":
            if options.params:
                encoded_params = urlencode(
                    [(k, v) for k, v in options.params.items() if v is not None]
                )
                abs_url = _build_api_url(abs_url, encoded_params)
        elif options.method.lower() in {"post", "put"}:
            if options.params and (options.files or options.override_headers):
                data = options.params
            elif options.params and not options.files:
                data_bytes = json.dumps(options.params).encode()
                headers["Content-Type"] = "application/json"

        else:
            raise error.APIConnectionError(
                "Unrecognized HTTP method %r. This may indicate a bug in the "
                "Together SDK. Please contact us by filling out https://www.together.ai/contact for "
                "assistance." % (options.method,)
            )

        if not options.override_headers:
            headers = utils.get_headers(options.method, self.api_key, headers)

        utils.log_debug(
            "Request to Together API",
            method=options.method,
            path=abs_url,
            post_data=(data or data_bytes),
            headers=json.dumps(headers),
        )

        return abs_url, headers, (data or data_bytes)

    def request_raw(
        self,
        options: TogetherRequest,
        remaining_retries: int,
        *,
        stream: bool = False,
        request_timeout: float | Tuple[float, float] | None = None,
        absolute: bool = False,
    ) -> requests.Response:
        abs_url, headers, data = self._prepare_request_raw(options, absolute)

        if not hasattr(_thread_context, "session"):
            _thread_context.session = _make_session(MAX_CONNECTION_RETRIES)
            _thread_context.session_create_time = time.time()
        elif (
            time.time() - getattr(_thread_context, "session_create_time", 0)
            >= MAX_SESSION_LIFETIME_SECS
        ):
            _thread_context.session.close()
            _thread_context.session = _make_session(MAX_CONNECTION_RETRIES)
            _thread_context.session_create_time = time.time()

        result = None
        try:
            result = _thread_context.session.request(
                options.method,
                abs_url,
                headers=headers,
                data=data,
                files=options.files,
                stream=stream,
                timeout=request_timeout or self.timeout,
                proxies=_thread_context.session.proxies,
                allow_redirects=options.allow_redirects,
            )
        except requests.exceptions.Timeout as e:
            utils.log_debug("Encountered requests.exceptions.Timeout")

            result_headers = dict(result.headers) if result is not None else {}

            if remaining_retries > 0:
                return self._retry_request(
                    options,
                    remaining_retries=remaining_retries,
                    response_headers=result_headers,
                    stream=stream,
                    request_timeout=request_timeout,
                )

            raise error.Timeout("Request timed out: {}".format(e)) from e
        except requests.exceptions.RequestException as e:
            utils.log_debug("Encountered requests.exceptions.RequestException")

            result_headers = dict(result.headers) if result is not None else {}

            if remaining_retries > 0:
                return self._retry_request(
                    options,
                    remaining_retries=remaining_retries,
                    response_headers=result_headers,
                    stream=stream,
                    request_timeout=request_timeout,
                )

            raise error.APIConnectionError(
                "Error communicating with API: {}".format(e)
            ) from e

        # retry on 5XX error or rate-limit
        if result is not None:
            if 500 <= result.status_code < 600 or result.status_code == 429:
                utils.log_debug(
                    f"Encountered requests.exceptions.HTTPError. Error code: {result.status_code}"
                )

                result_headers = dict(result.headers) if result is not None else {}

                if remaining_retries > 0:
                    return self._retry_request(
                        options,
                        remaining_retries=remaining_retries,
                        response_headers=result_headers,
                        stream=stream,
                        request_timeout=request_timeout,
                    )

        status_code = result.status_code if result is not None else 0
        result_headers = dict(result.headers) if result is not None else {}

        utils.log_debug(
            "Together API response",
            path=abs_url,
            response_code=status_code,
            processing_ms=result_headers.get("x-total-time"),
            request_id=result_headers.get("CF-RAY"),
        )

        return result  # type: ignore

    async def arequest_raw(
        self,
        options: TogetherRequest,
        session: aiohttp.ClientSession,
        *,
        request_timeout: float | Tuple[float, float] | None = None,
        absolute: bool = False,
    ) -> aiohttp.ClientResponse:
        abs_url, headers, data = self._prepare_request_raw(options, absolute)

        if isinstance(request_timeout, tuple):
            timeout = aiohttp.ClientTimeout(
                connect=request_timeout[0],
                total=request_timeout[1],
            )
        else:
            timeout = aiohttp.ClientTimeout(total=request_timeout or self.timeout)

        if options.files:
            data, content_type = requests.models.RequestEncodingMixin._encode_files(  # type: ignore
                options.files, data
            )
            headers["Content-Type"] = content_type

        request_kwargs = {
            "headers": headers,
            "data": data,
            "timeout": timeout,
            "allow_redirects": options.allow_redirects,
        }

        try:
            result = await session.request(
                method=options.method, url=abs_url, **request_kwargs
            )
            utils.log_debug(
                "Together API response",
                path=abs_url,
                response_code=result.status,
                processing_ms=result.headers.get("x-total-time"),
                request_id=result.headers.get("CF-RAY"),
            )
            # Don't read the whole stream for debug logging unless necessary.
            if together.log == "debug":
                utils.log_debug(
                    "API response body", body=result.content, headers=result.headers
                )
            return result
        except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
            raise error.Timeout("Request timed out") from e
        except aiohttp.ClientError as e:
            raise error.APIConnectionError("Error communicating with Together") from e

    def _interpret_response(
        self, result: requests.Response, stream: bool
    ) -> Tuple[TogetherResponse | Iterator[TogetherResponse], bool]:
        """Returns the response(s) and a bool indicating whether it is a stream."""
        if stream and "text/event-stream" in result.headers.get("Content-Type", ""):
            return (
                self._interpret_response_line(
                    line, result.status_code, result.headers, stream=True
                )
                for line in parse_stream(result.iter_lines())
            ), True
        else:
            return (
                self._interpret_response_line(
                    result.content.decode("utf-8"),
                    result.status_code,
                    result.headers,
                    stream=False,
                ),
                False,
            )

    async def _interpret_async_response(
        self, result: aiohttp.ClientResponse, stream: bool
    ) -> (
        tuple[AsyncGenerator[TogetherResponse, None], bool]
        | tuple[TogetherResponse, bool]
    ):
        """Returns the response(s) and a bool indicating whether it is a stream."""
        if stream and "text/event-stream" in result.headers.get("Content-Type", ""):
            return (
                self._interpret_response_line(
                    line, result.status, result.headers, stream=True
                )
                async for line in parse_stream_async(result.content)
            ), True
        else:
            try:
                await result.read()
            except (aiohttp.ServerTimeoutError, asyncio.TimeoutError) as e:
                raise error.Timeout("Request timed out") from e
            except aiohttp.ClientError as e:
                utils.log_warn(e, body=result.content)
            return (
                self._interpret_response_line(
                    (await result.read()).decode("utf-8"),
                    result.status,
                    result.headers,
                    stream=False,
                ),
                False,
            )

    def _interpret_response_line(
        self, rbody: str, rcode: int, rheaders: Any, stream: bool
    ) -> TogetherResponse:
        # HTTP 204 response code does not have any content in the body.
        if rcode == 204:
            return TogetherResponse({}, rheaders)

        if rcode == 503:
            raise error.ServiceUnavailableError(
                "The server is overloaded or not ready yet.",
                http_status=rcode,
                headers=rheaders,
            )

        try:
            if "text/plain" in rheaders.get("Content-Type", ""):
                data: Dict[str, Any] = {"message": rbody}
            else:
                data = json.loads(rbody)
        except (JSONDecodeError, UnicodeDecodeError) as e:
            raise error.APIError(
                f"Error code: {rcode} -{rbody}",
                http_status=rcode,
                headers=rheaders,
            ) from e
        resp = TogetherResponse(data, rheaders)

        # Handle streaming errors
        if not 200 <= rcode < 300:
            raise self.handle_error_response(resp, rcode, stream_error=stream)
        return resp


class AioHTTPSession(AsyncContextManager[aiohttp.ClientSession]):
    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._should_close_session: bool = False

    async def __aenter__(self) -> aiohttp.ClientSession:
        self._session = together.aiosession.get()
        if self._session is None:
            self._session = await aiohttp.ClientSession().__aenter__()
            self._should_close_session = True

        return self._session

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self._session is None:
            raise RuntimeError("Session is not initialized")

        if self._should_close_session:
            await self._session.__aexit__(exc_type, exc_value, traceback)
