import collections.abc
import enum
import modal._functions
import modal.functions
import modal_proto.api_pb2
import typing
import typing_extensions

class _PartialFunctionFlags(enum.IntFlag):
    FUNCTION: int = 1
    BUILD: int = 2
    ENTER_PRE_SNAPSHOT: int = 4
    ENTER_POST_SNAPSHOT: int = 8
    EXIT: int = 16
    BATCHED: int = 32
    CLUSTERED: int = 64  # Experimental: Clustered functions

    @staticmethod
    def all() -> int:
        return ~_PartialFunctionFlags(0)

P = typing_extensions.ParamSpec("P")

ReturnType = typing.TypeVar("ReturnType", covariant=True)

OriginalReturnType = typing.TypeVar("OriginalReturnType", covariant=True)

class _PartialFunction(typing.Generic[P, ReturnType, OriginalReturnType]):
    raw_f: collections.abc.Callable[P, ReturnType]
    flags: _PartialFunctionFlags
    webhook_config: typing.Optional[modal_proto.api_pb2.WebhookConfig]
    is_generator: bool
    keep_warm: typing.Optional[int]
    batch_max_size: typing.Optional[int]
    batch_wait_ms: typing.Optional[int]
    force_build: bool
    cluster_size: typing.Optional[int]
    build_timeout: typing.Optional[int]

    def __init__(
        self,
        raw_f: collections.abc.Callable[P, ReturnType],
        flags: _PartialFunctionFlags,
        webhook_config: typing.Optional[modal_proto.api_pb2.WebhookConfig] = None,
        is_generator: typing.Optional[bool] = None,
        keep_warm: typing.Optional[int] = None,
        batch_max_size: typing.Optional[int] = None,
        batch_wait_ms: typing.Optional[int] = None,
        cluster_size: typing.Optional[int] = None,
        force_build: bool = False,
        build_timeout: typing.Optional[int] = None,
    ): ...
    def _get_raw_f(self) -> collections.abc.Callable[P, ReturnType]: ...
    def _is_web_endpoint(self) -> bool: ...
    def __get__(self, obj, objtype=None) -> modal._functions._Function[P, ReturnType, OriginalReturnType]: ...
    def __del__(self): ...
    def add_flags(self, flags) -> _PartialFunction: ...

class PartialFunction(typing.Generic[P, ReturnType, OriginalReturnType]):
    raw_f: collections.abc.Callable[P, ReturnType]
    flags: _PartialFunctionFlags
    webhook_config: typing.Optional[modal_proto.api_pb2.WebhookConfig]
    is_generator: bool
    keep_warm: typing.Optional[int]
    batch_max_size: typing.Optional[int]
    batch_wait_ms: typing.Optional[int]
    force_build: bool
    cluster_size: typing.Optional[int]
    build_timeout: typing.Optional[int]

    def __init__(
        self,
        raw_f: collections.abc.Callable[P, ReturnType],
        flags: _PartialFunctionFlags,
        webhook_config: typing.Optional[modal_proto.api_pb2.WebhookConfig] = None,
        is_generator: typing.Optional[bool] = None,
        keep_warm: typing.Optional[int] = None,
        batch_max_size: typing.Optional[int] = None,
        batch_wait_ms: typing.Optional[int] = None,
        cluster_size: typing.Optional[int] = None,
        force_build: bool = False,
        build_timeout: typing.Optional[int] = None,
    ): ...
    def _get_raw_f(self) -> collections.abc.Callable[P, ReturnType]: ...
    def _is_web_endpoint(self) -> bool: ...
    def __get__(self, obj, objtype=None) -> modal.functions.Function[P, ReturnType, OriginalReturnType]: ...
    def __del__(self): ...
    def add_flags(self, flags) -> PartialFunction: ...

def _find_partial_methods_for_user_cls(user_cls: type[typing.Any], flags: int) -> dict[str, _PartialFunction]: ...
def _find_callables_for_obj(
    user_obj: typing.Any, flags: int
) -> dict[str, collections.abc.Callable[..., typing.Any]]: ...

class _MethodDecoratorType:
    @typing.overload
    def __call__(
        self, func: PartialFunction[typing_extensions.Concatenate[typing.Any, P], ReturnType, OriginalReturnType]
    ) -> PartialFunction[P, ReturnType, OriginalReturnType]: ...
    @typing.overload
    def __call__(
        self,
        func: collections.abc.Callable[
            typing_extensions.Concatenate[typing.Any, P], collections.abc.Coroutine[typing.Any, typing.Any, ReturnType]
        ],
    ) -> PartialFunction[P, ReturnType, collections.abc.Coroutine[typing.Any, typing.Any, ReturnType]]: ...
    @typing.overload
    def __call__(
        self, func: collections.abc.Callable[typing_extensions.Concatenate[typing.Any, P], ReturnType]
    ) -> PartialFunction[P, ReturnType, ReturnType]: ...

def _method(
    _warn_parentheses_missing=None,
    *,
    is_generator: typing.Optional[bool] = None,
    keep_warm: typing.Optional[int] = None,
) -> _MethodDecoratorType: ...
def _parse_custom_domains(
    custom_domains: typing.Optional[collections.abc.Iterable[str]] = None,
) -> list[modal_proto.api_pb2.CustomDomainConfig]: ...
def _web_endpoint(
    _warn_parentheses_missing=None,
    *,
    method: str = "GET",
    label: typing.Optional[str] = None,
    docs: bool = False,
    custom_domains: typing.Optional[collections.abc.Iterable[str]] = None,
    requires_proxy_auth: bool = False,
    wait_for_response: bool = True,
) -> collections.abc.Callable[
    [collections.abc.Callable[P, ReturnType]], _PartialFunction[P, ReturnType, ReturnType]
]: ...
def _asgi_app(
    _warn_parentheses_missing=None,
    *,
    label: typing.Optional[str] = None,
    custom_domains: typing.Optional[collections.abc.Iterable[str]] = None,
    requires_proxy_auth: bool = False,
    wait_for_response: bool = True,
) -> collections.abc.Callable[[collections.abc.Callable[..., typing.Any]], _PartialFunction]: ...
def _wsgi_app(
    _warn_parentheses_missing=None,
    *,
    label: typing.Optional[str] = None,
    custom_domains: typing.Optional[collections.abc.Iterable[str]] = None,
    requires_proxy_auth: bool = False,
    wait_for_response: bool = True,
) -> collections.abc.Callable[[collections.abc.Callable[..., typing.Any]], _PartialFunction]: ...
def _web_server(
    port: int,
    *,
    startup_timeout: float = 5.0,
    label: typing.Optional[str] = None,
    custom_domains: typing.Optional[collections.abc.Iterable[str]] = None,
    requires_proxy_auth: bool = False,
) -> collections.abc.Callable[[collections.abc.Callable[..., typing.Any]], _PartialFunction]: ...
def _disallow_wrapping_method(f: _PartialFunction, wrapper: str) -> None: ...
def _build(
    _warn_parentheses_missing=None, *, force: bool = False, timeout: int = 86400
) -> collections.abc.Callable[
    [typing.Union[collections.abc.Callable[[typing.Any], typing.Any], _PartialFunction]], _PartialFunction
]: ...
def _enter(
    _warn_parentheses_missing=None, *, snap: bool = False
) -> collections.abc.Callable[
    [typing.Union[collections.abc.Callable[[typing.Any], typing.Any], _PartialFunction]], _PartialFunction
]: ...
def _exit(
    _warn_parentheses_missing=None,
) -> collections.abc.Callable[
    [
        typing.Union[
            collections.abc.Callable[
                [typing.Any, typing.Optional[type[BaseException]], typing.Optional[BaseException], typing.Any],
                typing.Any,
            ],
            collections.abc.Callable[[typing.Any], typing.Any],
        ]
    ],
    _PartialFunction,
]: ...
def _batched(
    _warn_parentheses_missing=None, *, max_batch_size: int, wait_ms: int
) -> collections.abc.Callable[[collections.abc.Callable[..., typing.Any]], _PartialFunction]: ...
def method(
    _warn_parentheses_missing=None,
    *,
    is_generator: typing.Optional[bool] = None,
    keep_warm: typing.Optional[int] = None,
) -> _MethodDecoratorType: ...
def web_endpoint(
    _warn_parentheses_missing=None,
    *,
    method: str = "GET",
    label: typing.Optional[str] = None,
    docs: bool = False,
    custom_domains: typing.Optional[collections.abc.Iterable[str]] = None,
    requires_proxy_auth: bool = False,
    wait_for_response: bool = True,
) -> collections.abc.Callable[
    [collections.abc.Callable[P, ReturnType]], PartialFunction[P, ReturnType, ReturnType]
]: ...
def asgi_app(
    _warn_parentheses_missing=None,
    *,
    label: typing.Optional[str] = None,
    custom_domains: typing.Optional[collections.abc.Iterable[str]] = None,
    requires_proxy_auth: bool = False,
    wait_for_response: bool = True,
) -> collections.abc.Callable[[collections.abc.Callable[..., typing.Any]], PartialFunction]: ...
def wsgi_app(
    _warn_parentheses_missing=None,
    *,
    label: typing.Optional[str] = None,
    custom_domains: typing.Optional[collections.abc.Iterable[str]] = None,
    requires_proxy_auth: bool = False,
    wait_for_response: bool = True,
) -> collections.abc.Callable[[collections.abc.Callable[..., typing.Any]], PartialFunction]: ...
def web_server(
    port: int,
    *,
    startup_timeout: float = 5.0,
    label: typing.Optional[str] = None,
    custom_domains: typing.Optional[collections.abc.Iterable[str]] = None,
    requires_proxy_auth: bool = False,
) -> collections.abc.Callable[[collections.abc.Callable[..., typing.Any]], PartialFunction]: ...
def build(
    _warn_parentheses_missing=None, *, force: bool = False, timeout: int = 86400
) -> collections.abc.Callable[
    [typing.Union[collections.abc.Callable[[typing.Any], typing.Any], PartialFunction]], PartialFunction
]: ...
def enter(
    _warn_parentheses_missing=None, *, snap: bool = False
) -> collections.abc.Callable[
    [typing.Union[collections.abc.Callable[[typing.Any], typing.Any], PartialFunction]], PartialFunction
]: ...
def exit(
    _warn_parentheses_missing=None,
) -> collections.abc.Callable[
    [
        typing.Union[
            collections.abc.Callable[
                [typing.Any, typing.Optional[type[BaseException]], typing.Optional[BaseException], typing.Any],
                typing.Any,
            ],
            collections.abc.Callable[[typing.Any], typing.Any],
        ]
    ],
    PartialFunction,
]: ...
def batched(
    _warn_parentheses_missing=None, *, max_batch_size: int, wait_ms: int
) -> collections.abc.Callable[[collections.abc.Callable[..., typing.Any]], PartialFunction]: ...
