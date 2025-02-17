import collections.abc
import google.protobuf.message
import modal._functions
import modal._utils.async_utils
import modal._utils.function_utils
import modal.app
import modal.call_graph
import modal.client
import modal.cloud_bucket_mount
import modal.cls
import modal.gpu
import modal.image
import modal.mount
import modal.network_file_system
import modal.object
import modal.parallel_map
import modal.partial_function
import modal.proxy
import modal.retries
import modal.schedule
import modal.scheduler_placement
import modal.secret
import modal.volume
import modal_proto.api_pb2
import pathlib
import typing
import typing_extensions

SUPERSELF = typing.TypeVar("SUPERSELF", covariant=True)

ReturnType_INNER = typing.TypeVar("ReturnType_INNER", covariant=True)

P_INNER = typing_extensions.ParamSpec("P_INNER")

class Function(
    typing.Generic[modal._functions.P, modal._functions.ReturnType, modal._functions.OriginalReturnType],
    modal.object.Object,
):
    _info: typing.Optional[modal._utils.function_utils.FunctionInfo]
    _serve_mounts: frozenset[modal.mount.Mount]
    _app: typing.Optional[modal.app.App]
    _obj: typing.Optional[modal.cls.Obj]
    _webhook_config: typing.Optional[modal_proto.api_pb2.WebhookConfig]
    _web_url: typing.Optional[str]
    _function_name: typing.Optional[str]
    _is_method: bool
    _spec: typing.Optional[modal._functions._FunctionSpec]
    _tag: str
    _raw_f: typing.Optional[collections.abc.Callable[..., typing.Any]]
    _build_args: dict
    _is_generator: typing.Optional[bool]
    _cluster_size: typing.Optional[int]
    _use_method_name: str
    _class_parameter_info: typing.Optional[modal_proto.api_pb2.ClassParameterInfo]
    _method_handle_metadata: typing.Optional[dict[str, modal_proto.api_pb2.FunctionHandleMetadata]]

    def __init__(self, *args, **kwargs): ...
    def _bind_method(self, user_cls, method_name: str, partial_function: modal.partial_function.PartialFunction): ...
    @staticmethod
    def from_args(
        info: modal._utils.function_utils.FunctionInfo,
        app,
        image: modal.image.Image,
        secrets: collections.abc.Sequence[modal.secret.Secret] = (),
        schedule: typing.Optional[modal.schedule.Schedule] = None,
        is_generator: bool = False,
        gpu: typing.Union[
            None, bool, str, modal.gpu._GPUConfig, list[typing.Union[None, bool, str, modal.gpu._GPUConfig]]
        ] = None,
        mounts: collections.abc.Collection[modal.mount.Mount] = (),
        network_file_systems: dict[
            typing.Union[str, pathlib.PurePosixPath], modal.network_file_system.NetworkFileSystem
        ] = {},
        allow_cross_region_volumes: bool = False,
        volumes: dict[
            typing.Union[str, pathlib.PurePosixPath],
            typing.Union[modal.volume.Volume, modal.cloud_bucket_mount.CloudBucketMount],
        ] = {},
        webhook_config: typing.Optional[modal_proto.api_pb2.WebhookConfig] = None,
        memory: typing.Union[int, tuple[int, int], None] = None,
        proxy: typing.Optional[modal.proxy.Proxy] = None,
        retries: typing.Union[int, modal.retries.Retries, None] = None,
        timeout: typing.Optional[int] = None,
        concurrency_limit: typing.Optional[int] = None,
        allow_concurrent_inputs: typing.Optional[int] = None,
        batch_max_size: typing.Optional[int] = None,
        batch_wait_ms: typing.Optional[int] = None,
        container_idle_timeout: typing.Optional[int] = None,
        cpu: typing.Union[float, tuple[float, float], None] = None,
        keep_warm: typing.Optional[int] = None,
        cloud: typing.Optional[str] = None,
        scheduler_placement: typing.Optional[modal.scheduler_placement.SchedulerPlacement] = None,
        is_builder_function: bool = False,
        is_auto_snapshot: bool = False,
        enable_memory_snapshot: bool = False,
        block_network: bool = False,
        i6pn_enabled: bool = False,
        cluster_size: typing.Optional[int] = None,
        max_inputs: typing.Optional[int] = None,
        ephemeral_disk: typing.Optional[int] = None,
        include_source: typing.Optional[bool] = None,
        _experimental_buffer_containers: typing.Optional[int] = None,
        _experimental_proxy_ip: typing.Optional[str] = None,
        _experimental_custom_scaling_factor: typing.Optional[float] = None,
    ) -> Function: ...
    def _bind_parameters(
        self,
        obj: modal.cls.Obj,
        options: typing.Optional[modal_proto.api_pb2.FunctionOptions],
        args: collections.abc.Sized,
        kwargs: dict[str, typing.Any],
    ) -> Function: ...

    class __keep_warm_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, warm_pool_size: int) -> None: ...
        async def aio(self, warm_pool_size: int) -> None: ...

    keep_warm: __keep_warm_spec[typing_extensions.Self]

    @classmethod
    def from_name(
        cls: type[Function], app_name: str, name: str, namespace=1, environment_name: typing.Optional[str] = None
    ) -> Function: ...

    class __lookup_spec(typing_extensions.Protocol):
        def __call__(
            self,
            app_name: str,
            name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
        ) -> Function: ...
        async def aio(
            self,
            app_name: str,
            name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
        ) -> Function: ...

    lookup: __lookup_spec

    @property
    def tag(self) -> str: ...
    @property
    def app(self) -> modal.app.App: ...
    @property
    def stub(self) -> modal.app.App: ...
    @property
    def info(self) -> modal._utils.function_utils.FunctionInfo: ...
    @property
    def spec(self) -> modal._functions._FunctionSpec: ...
    def _is_web_endpoint(self) -> bool: ...
    def get_build_def(self) -> str: ...
    def _initialize_from_empty(self): ...
    def _hydrate_metadata(self, metadata: typing.Optional[google.protobuf.message.Message]): ...
    def _get_metadata(self): ...
    def _check_no_web_url(self, fn_name: str): ...
    @property
    def web_url(self) -> str: ...
    @property
    def is_generator(self) -> bool: ...
    @property
    def cluster_size(self) -> int: ...

    class ___map_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self, input_queue: modal.parallel_map.SynchronizedQueue, order_outputs: bool, return_exceptions: bool
        ) -> typing.Generator[typing.Any, None, None]: ...
        def aio(
            self, input_queue: modal.parallel_map.SynchronizedQueue, order_outputs: bool, return_exceptions: bool
        ) -> collections.abc.AsyncGenerator[typing.Any, None]: ...

    _map: ___map_spec[typing_extensions.Self]

    class ___call_function_spec(typing_extensions.Protocol[ReturnType_INNER, SUPERSELF]):
        def __call__(self, args, kwargs) -> ReturnType_INNER: ...
        async def aio(self, args, kwargs) -> ReturnType_INNER: ...

    _call_function: ___call_function_spec[modal._functions.ReturnType, typing_extensions.Self]

    class ___call_function_nowait_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, args, kwargs, function_call_invocation_type: int) -> modal._functions._Invocation: ...
        async def aio(self, args, kwargs, function_call_invocation_type: int) -> modal._functions._Invocation: ...

    _call_function_nowait: ___call_function_nowait_spec[typing_extensions.Self]

    class ___call_generator_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, args, kwargs): ...
        def aio(self, args, kwargs): ...

    _call_generator: ___call_generator_spec[typing_extensions.Self]

    class ___call_generator_nowait_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, args, kwargs): ...
        async def aio(self, args, kwargs): ...

    _call_generator_nowait: ___call_generator_nowait_spec[typing_extensions.Self]

    class __remote_spec(typing_extensions.Protocol[P_INNER, ReturnType_INNER, SUPERSELF]):
        def __call__(self, *args: P_INNER.args, **kwargs: P_INNER.kwargs) -> ReturnType_INNER: ...
        async def aio(self, *args: P_INNER.args, **kwargs: P_INNER.kwargs) -> ReturnType_INNER: ...

    remote: __remote_spec[modal._functions.P, modal._functions.ReturnType, typing_extensions.Self]

    class __remote_gen_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, *args, **kwargs) -> typing.Generator[typing.Any, None, None]: ...
        def aio(self, *args, **kwargs) -> collections.abc.AsyncGenerator[typing.Any, None]: ...

    remote_gen: __remote_gen_spec[typing_extensions.Self]

    def _is_local(self): ...
    def _get_info(self) -> modal._utils.function_utils.FunctionInfo: ...
    def _get_obj(self) -> typing.Optional[modal.cls.Obj]: ...
    def local(
        self, *args: modal._functions.P.args, **kwargs: modal._functions.P.kwargs
    ) -> modal._functions.OriginalReturnType: ...

    class ___experimental_spawn_spec(typing_extensions.Protocol[P_INNER, ReturnType_INNER, SUPERSELF]):
        def __call__(self, *args: P_INNER.args, **kwargs: P_INNER.kwargs) -> FunctionCall[ReturnType_INNER]: ...
        async def aio(self, *args: P_INNER.args, **kwargs: P_INNER.kwargs) -> FunctionCall[ReturnType_INNER]: ...

    _experimental_spawn: ___experimental_spawn_spec[
        modal._functions.P, modal._functions.ReturnType, typing_extensions.Self
    ]

    class __spawn_spec(typing_extensions.Protocol[P_INNER, ReturnType_INNER, SUPERSELF]):
        def __call__(self, *args: P_INNER.args, **kwargs: P_INNER.kwargs) -> FunctionCall[ReturnType_INNER]: ...
        async def aio(self, *args: P_INNER.args, **kwargs: P_INNER.kwargs) -> FunctionCall[ReturnType_INNER]: ...

    spawn: __spawn_spec[modal._functions.P, modal._functions.ReturnType, typing_extensions.Self]

    def get_raw_f(self) -> collections.abc.Callable[..., typing.Any]: ...

    class __get_current_stats_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self) -> modal._functions.FunctionStats: ...
        async def aio(self) -> modal._functions.FunctionStats: ...

    get_current_stats: __get_current_stats_spec[typing_extensions.Self]

    class __map_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self, *input_iterators, kwargs={}, order_outputs: bool = True, return_exceptions: bool = False
        ) -> modal._utils.async_utils.AsyncOrSyncIterable: ...
        def aio(
            self,
            *input_iterators: typing.Union[typing.Iterable[typing.Any], typing.AsyncIterable[typing.Any]],
            kwargs={},
            order_outputs: bool = True,
            return_exceptions: bool = False,
        ) -> typing.AsyncGenerator[typing.Any, None]: ...

    map: __map_spec[typing_extensions.Self]

    class __starmap_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self,
            input_iterator: typing.Iterable[typing.Sequence[typing.Any]],
            kwargs={},
            order_outputs: bool = True,
            return_exceptions: bool = False,
        ) -> modal._utils.async_utils.AsyncOrSyncIterable: ...
        def aio(
            self,
            input_iterator: typing.Union[
                typing.Iterable[typing.Sequence[typing.Any]], typing.AsyncIterable[typing.Sequence[typing.Any]]
            ],
            kwargs={},
            order_outputs: bool = True,
            return_exceptions: bool = False,
        ) -> typing.AsyncIterable[typing.Any]: ...

    starmap: __starmap_spec[typing_extensions.Self]

    class __for_each_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, *input_iterators, kwargs={}, ignore_exceptions: bool = False): ...
        async def aio(self, *input_iterators, kwargs={}, ignore_exceptions: bool = False): ...

    for_each: __for_each_spec[typing_extensions.Self]

class FunctionCall(typing.Generic[modal._functions.ReturnType], modal.object.Object):
    _is_generator: bool

    def __init__(self, *args, **kwargs): ...
    def _invocation(self): ...

    class __get_spec(typing_extensions.Protocol[ReturnType_INNER, SUPERSELF]):
        def __call__(self, timeout: typing.Optional[float] = None) -> ReturnType_INNER: ...
        async def aio(self, timeout: typing.Optional[float] = None) -> ReturnType_INNER: ...

    get: __get_spec[modal._functions.ReturnType, typing_extensions.Self]

    class __get_gen_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self) -> typing.Generator[typing.Any, None, None]: ...
        def aio(self) -> collections.abc.AsyncGenerator[typing.Any, None]: ...

    get_gen: __get_gen_spec[typing_extensions.Self]

    class __get_call_graph_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self) -> list[modal.call_graph.InputInfo]: ...
        async def aio(self) -> list[modal.call_graph.InputInfo]: ...

    get_call_graph: __get_call_graph_spec[typing_extensions.Self]

    class __cancel_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, terminate_containers: bool = False): ...
        async def aio(self, terminate_containers: bool = False): ...

    cancel: __cancel_spec[typing_extensions.Self]

    class __from_id_spec(typing_extensions.Protocol):
        def __call__(
            self, function_call_id: str, client: typing.Optional[modal.client.Client] = None, is_generator: bool = False
        ) -> FunctionCall[typing.Any]: ...
        async def aio(
            self, function_call_id: str, client: typing.Optional[modal.client.Client] = None, is_generator: bool = False
        ) -> FunctionCall[typing.Any]: ...

    from_id: __from_id_spec

class __gather_spec(typing_extensions.Protocol):
    def __call__(
        self, *function_calls: FunctionCall[modal._functions.ReturnType]
    ) -> typing.Sequence[modal._functions.ReturnType]: ...
    async def aio(
        self, *function_calls: FunctionCall[modal._functions.ReturnType]
    ) -> typing.Sequence[modal._functions.ReturnType]: ...

gather: __gather_spec
