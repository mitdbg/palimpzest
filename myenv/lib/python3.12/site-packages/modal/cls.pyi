import collections.abc
import google.protobuf.message
import inspect
import modal._functions
import modal._object
import modal.app
import modal.client
import modal.functions
import modal.gpu
import modal.object
import modal.partial_function
import modal.retries
import modal.secret
import modal.volume
import modal_proto.api_pb2
import os
import typing
import typing_extensions

T = typing.TypeVar("T")

def _use_annotation_parameters(user_cls: type) -> bool: ...
def _get_class_constructor_signature(user_cls: type) -> inspect.Signature: ...
def _bind_instance_method(
    service_function: modal._functions._Function, class_bound_method: modal._functions._Function
): ...

class _Obj:
    _cls: _Cls
    _functions: dict[str, modal._functions._Function]
    _has_entered: bool
    _user_cls_instance: typing.Optional[typing.Any]
    _args: tuple[typing.Any, ...]
    _kwargs: dict[str, typing.Any]
    _instance_service_function: typing.Optional[modal._functions._Function]

    def _uses_common_service_function(self): ...
    def __init__(
        self,
        cls: _Cls,
        user_cls: typing.Optional[type],
        options: typing.Optional[modal_proto.api_pb2.FunctionOptions],
        args,
        kwargs,
    ): ...
    def _cached_service_function(self) -> modal._functions._Function: ...
    def _get_parameter_values(self) -> dict[str, typing.Any]: ...
    def _new_user_cls_instance(self): ...
    async def keep_warm(self, warm_pool_size: int) -> None: ...
    def _cached_user_cls_instance(self): ...
    def _enter(self): ...
    @property
    def _entered(self) -> bool: ...
    @_entered.setter
    def _entered(self, val: bool): ...
    async def _aenter(self): ...
    def __getattr__(self, k): ...

SUPERSELF = typing.TypeVar("SUPERSELF", covariant=True)

class Obj:
    _cls: Cls
    _functions: dict[str, modal.functions.Function]
    _has_entered: bool
    _user_cls_instance: typing.Optional[typing.Any]
    _args: tuple[typing.Any, ...]
    _kwargs: dict[str, typing.Any]
    _instance_service_function: typing.Optional[modal.functions.Function]

    def __init__(
        self,
        cls: Cls,
        user_cls: typing.Optional[type],
        options: typing.Optional[modal_proto.api_pb2.FunctionOptions],
        args,
        kwargs,
    ): ...
    def _uses_common_service_function(self): ...
    def _cached_service_function(self) -> modal.functions.Function: ...
    def _get_parameter_values(self) -> dict[str, typing.Any]: ...
    def _new_user_cls_instance(self): ...

    class __keep_warm_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, warm_pool_size: int) -> None: ...
        async def aio(self, warm_pool_size: int) -> None: ...

    keep_warm: __keep_warm_spec[typing_extensions.Self]

    def _cached_user_cls_instance(self): ...
    def _enter(self): ...
    @property
    def _entered(self) -> bool: ...
    @_entered.setter
    def _entered(self, val: bool): ...
    async def _aenter(self): ...
    def __getattr__(self, k): ...

class _Cls(modal._object._Object):
    _user_cls: typing.Optional[type]
    _class_service_function: typing.Optional[modal._functions._Function]
    _method_functions: typing.Optional[dict[str, modal._functions._Function]]
    _options: typing.Optional[modal_proto.api_pb2.FunctionOptions]
    _callables: dict[str, collections.abc.Callable[..., typing.Any]]
    _app: typing.Optional[modal.app._App]
    _name: typing.Optional[str]

    def _initialize_from_empty(self): ...
    def _initialize_from_other(self, other: _Cls): ...
    def _get_partial_functions(self) -> dict[str, modal.partial_function._PartialFunction]: ...
    def _get_app(self) -> modal.app._App: ...
    def _get_user_cls(self) -> type: ...
    def _get_name(self) -> str: ...
    def _get_class_service_function(self) -> modal._functions._Function: ...
    def _get_method_names(self) -> collections.abc.Collection[str]: ...
    def _hydrate_metadata(self, metadata: google.protobuf.message.Message): ...
    @staticmethod
    def validate_construction_mechanism(user_cls): ...
    @staticmethod
    def from_local(user_cls, app: modal.app._App, class_service_function: modal._functions._Function) -> _Cls: ...
    def _uses_common_service_function(self): ...
    @classmethod
    def from_name(
        cls: type[_Cls],
        app_name: str,
        name: str,
        namespace=1,
        environment_name: typing.Optional[str] = None,
        workspace: typing.Optional[str] = None,
    ) -> _Cls: ...
    def with_options(
        self: _Cls,
        cpu: typing.Union[float, tuple[float, float], None] = None,
        memory: typing.Union[int, tuple[int, int], None] = None,
        gpu: typing.Union[None, bool, str, modal.gpu._GPUConfig] = None,
        secrets: collections.abc.Collection[modal.secret._Secret] = (),
        volumes: dict[typing.Union[str, os.PathLike], modal.volume._Volume] = {},
        retries: typing.Union[int, modal.retries.Retries, None] = None,
        timeout: typing.Optional[int] = None,
        concurrency_limit: typing.Optional[int] = None,
        allow_concurrent_inputs: typing.Optional[int] = None,
        container_idle_timeout: typing.Optional[int] = None,
    ) -> _Cls: ...
    @staticmethod
    async def lookup(
        app_name: str,
        name: str,
        namespace=1,
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
        workspace: typing.Optional[str] = None,
    ) -> _Cls: ...
    def __call__(self, *args, **kwargs) -> _Obj: ...
    def __getattr__(self, k): ...
    def _is_local(self) -> bool: ...

class Cls(modal.object.Object):
    _user_cls: typing.Optional[type]
    _class_service_function: typing.Optional[modal.functions.Function]
    _method_functions: typing.Optional[dict[str, modal.functions.Function]]
    _options: typing.Optional[modal_proto.api_pb2.FunctionOptions]
    _callables: dict[str, collections.abc.Callable[..., typing.Any]]
    _app: typing.Optional[modal.app.App]
    _name: typing.Optional[str]

    def __init__(self, *args, **kwargs): ...
    def _initialize_from_empty(self): ...
    def _initialize_from_other(self, other: Cls): ...
    def _get_partial_functions(self) -> dict[str, modal.partial_function.PartialFunction]: ...
    def _get_app(self) -> modal.app.App: ...
    def _get_user_cls(self) -> type: ...
    def _get_name(self) -> str: ...
    def _get_class_service_function(self) -> modal.functions.Function: ...
    def _get_method_names(self) -> collections.abc.Collection[str]: ...
    def _hydrate_metadata(self, metadata: google.protobuf.message.Message): ...
    @staticmethod
    def validate_construction_mechanism(user_cls): ...
    @staticmethod
    def from_local(user_cls, app: modal.app.App, class_service_function: modal.functions.Function) -> Cls: ...
    def _uses_common_service_function(self): ...
    @classmethod
    def from_name(
        cls: type[Cls],
        app_name: str,
        name: str,
        namespace=1,
        environment_name: typing.Optional[str] = None,
        workspace: typing.Optional[str] = None,
    ) -> Cls: ...
    def with_options(
        self: Cls,
        cpu: typing.Union[float, tuple[float, float], None] = None,
        memory: typing.Union[int, tuple[int, int], None] = None,
        gpu: typing.Union[None, bool, str, modal.gpu._GPUConfig] = None,
        secrets: collections.abc.Collection[modal.secret.Secret] = (),
        volumes: dict[typing.Union[str, os.PathLike], modal.volume.Volume] = {},
        retries: typing.Union[int, modal.retries.Retries, None] = None,
        timeout: typing.Optional[int] = None,
        concurrency_limit: typing.Optional[int] = None,
        allow_concurrent_inputs: typing.Optional[int] = None,
        container_idle_timeout: typing.Optional[int] = None,
    ) -> Cls: ...

    class __lookup_spec(typing_extensions.Protocol):
        def __call__(
            self,
            app_name: str,
            name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            workspace: typing.Optional[str] = None,
        ) -> Cls: ...
        async def aio(
            self,
            app_name: str,
            name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            workspace: typing.Optional[str] = None,
        ) -> Cls: ...

    lookup: __lookup_spec

    def __call__(self, *args, **kwargs) -> Obj: ...
    def __getattr__(self, k): ...
    def _is_local(self) -> bool: ...

class _NO_DEFAULT:
    def __repr__(self): ...

_no_default: _NO_DEFAULT

class _Parameter:
    default: typing.Any
    init: bool

    def __init__(self, default: typing.Any, init: bool): ...
    def __get__(self, obj, obj_type=None) -> typing.Any: ...

def is_parameter(p: typing.Any) -> bool: ...
def parameter(*, default: typing.Any = modal.cls._NO_DEFAULT(), init: bool = True) -> typing.Any: ...
