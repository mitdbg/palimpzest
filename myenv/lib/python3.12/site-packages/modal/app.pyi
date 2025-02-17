import collections.abc
import modal._functions
import modal._object
import modal._utils.function_utils
import modal.client
import modal.cloud_bucket_mount
import modal.cls
import modal.functions
import modal.gpu
import modal.image
import modal.mount
import modal.network_file_system
import modal.object
import modal.partial_function
import modal.proxy
import modal.retries
import modal.running_app
import modal.schedule
import modal.scheduler_placement
import modal.secret
import modal.volume
import modal_proto.api_pb2
import pathlib
import synchronicity.combined_types
import typing
import typing_extensions

class _LocalEntrypoint:
    _info: modal._utils.function_utils.FunctionInfo
    _app: _App

    def __init__(self, info: modal._utils.function_utils.FunctionInfo, app: _App) -> None: ...
    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any: ...
    @property
    def info(self) -> modal._utils.function_utils.FunctionInfo: ...
    @property
    def app(self) -> _App: ...
    @property
    def stub(self) -> _App: ...

class LocalEntrypoint:
    _info: modal._utils.function_utils.FunctionInfo
    _app: App

    def __init__(self, info: modal._utils.function_utils.FunctionInfo, app: App) -> None: ...
    def __call__(self, *args: typing.Any, **kwargs: typing.Any) -> typing.Any: ...
    @property
    def info(self) -> modal._utils.function_utils.FunctionInfo: ...
    @property
    def app(self) -> App: ...
    @property
    def stub(self) -> App: ...

def check_sequence(items: typing.Sequence[typing.Any], item_type: type[typing.Any], error_msg: str) -> None: ...

CLS_T = typing.TypeVar("CLS_T", bound="type[typing.Any]")

P = typing_extensions.ParamSpec("P")

ReturnType = typing.TypeVar("ReturnType")

OriginalReturnType = typing.TypeVar("OriginalReturnType")

class _FunctionDecoratorType:
    @typing.overload
    def __call__(
        self, func: modal.partial_function.PartialFunction[P, ReturnType, OriginalReturnType]
    ) -> modal.functions.Function[P, ReturnType, OriginalReturnType]: ...
    @typing.overload
    def __call__(
        self, func: collections.abc.Callable[P, collections.abc.Coroutine[typing.Any, typing.Any, ReturnType]]
    ) -> modal.functions.Function[P, ReturnType, collections.abc.Coroutine[typing.Any, typing.Any, ReturnType]]: ...
    @typing.overload
    def __call__(
        self, func: collections.abc.Callable[P, ReturnType]
    ) -> modal.functions.Function[P, ReturnType, ReturnType]: ...

class _App:
    _all_apps: typing.ClassVar[dict[typing.Optional[str], list[_App]]]
    _container_app: typing.ClassVar[typing.Optional[_App]]
    _name: typing.Optional[str]
    _description: typing.Optional[str]
    _functions: dict[str, modal._functions._Function]
    _classes: dict[str, modal.cls._Cls]
    _image: typing.Optional[modal.image._Image]
    _mounts: collections.abc.Sequence[modal.mount._Mount]
    _secrets: collections.abc.Sequence[modal.secret._Secret]
    _volumes: dict[typing.Union[str, pathlib.PurePosixPath], modal.volume._Volume]
    _web_endpoints: list[str]
    _local_entrypoints: dict[str, _LocalEntrypoint]
    _app_id: typing.Optional[str]
    _running_app: typing.Optional[modal.running_app.RunningApp]
    _client: typing.Optional[modal.client._Client]
    _include_source_default: typing.Optional[bool]

    def __init__(
        self,
        name: typing.Optional[str] = None,
        *,
        image: typing.Optional[modal.image._Image] = None,
        mounts: collections.abc.Sequence[modal.mount._Mount] = [],
        secrets: collections.abc.Sequence[modal.secret._Secret] = [],
        volumes: dict[typing.Union[str, pathlib.PurePosixPath], modal.volume._Volume] = {},
        include_source: typing.Optional[bool] = None,
    ) -> None: ...
    @property
    def name(self) -> typing.Optional[str]: ...
    @property
    def is_interactive(self) -> bool: ...
    @property
    def app_id(self) -> typing.Optional[str]: ...
    @property
    def description(self) -> typing.Optional[str]: ...
    @staticmethod
    async def lookup(
        name: str,
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
        create_if_missing: bool = False,
    ) -> _App: ...
    def set_description(self, description: str): ...
    def _validate_blueprint_value(self, key: str, value: typing.Any): ...
    @property
    def image(self) -> modal.image._Image: ...
    @image.setter
    def image(self, value): ...
    def _uncreate_all_objects(self): ...
    def _set_local_app(
        self, client: modal.client._Client, running_app: modal.running_app.RunningApp
    ) -> typing.AsyncContextManager[None]: ...
    def run(
        self,
        client: typing.Optional[modal.client._Client] = None,
        show_progress: typing.Optional[bool] = None,
        detach: bool = False,
        interactive: bool = False,
        environment_name: typing.Optional[str] = None,
    ) -> typing.AsyncContextManager[_App]: ...
    def _get_default_image(self): ...
    def _get_watch_mounts(self): ...
    def _add_function(self, function: modal._functions._Function, is_web_endpoint: bool): ...
    def _add_class(self, tag: str, cls: modal.cls._Cls): ...
    def _init_container(self, client: modal.client._Client, running_app: modal.running_app.RunningApp): ...
    @property
    def registered_functions(self) -> dict[str, modal._functions._Function]: ...
    @property
    def registered_classes(self) -> dict[str, modal.cls._Cls]: ...
    @property
    def registered_entrypoints(self) -> dict[str, _LocalEntrypoint]: ...
    @property
    def indexed_objects(self) -> dict[str, modal._object._Object]: ...
    @property
    def registered_web_endpoints(self) -> list[str]: ...
    def local_entrypoint(
        self, _warn_parentheses_missing: typing.Any = None, *, name: typing.Optional[str] = None
    ) -> collections.abc.Callable[[collections.abc.Callable[..., typing.Any]], _LocalEntrypoint]: ...
    def function(
        self,
        _warn_parentheses_missing: typing.Any = None,
        *,
        image: typing.Optional[modal.image._Image] = None,
        schedule: typing.Optional[modal.schedule.Schedule] = None,
        secrets: collections.abc.Sequence[modal.secret._Secret] = (),
        gpu: typing.Union[
            None, bool, str, modal.gpu._GPUConfig, list[typing.Union[None, bool, str, modal.gpu._GPUConfig]]
        ] = None,
        serialized: bool = False,
        mounts: collections.abc.Sequence[modal.mount._Mount] = (),
        network_file_systems: dict[
            typing.Union[str, pathlib.PurePosixPath], modal.network_file_system._NetworkFileSystem
        ] = {},
        volumes: dict[
            typing.Union[str, pathlib.PurePosixPath],
            typing.Union[modal.volume._Volume, modal.cloud_bucket_mount._CloudBucketMount],
        ] = {},
        allow_cross_region_volumes: bool = False,
        cpu: typing.Union[float, tuple[float, float], None] = None,
        memory: typing.Union[int, tuple[int, int], None] = None,
        ephemeral_disk: typing.Optional[int] = None,
        proxy: typing.Optional[modal.proxy._Proxy] = None,
        retries: typing.Union[int, modal.retries.Retries, None] = None,
        concurrency_limit: typing.Optional[int] = None,
        allow_concurrent_inputs: typing.Optional[int] = None,
        container_idle_timeout: typing.Optional[int] = None,
        timeout: typing.Optional[int] = None,
        keep_warm: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
        is_generator: typing.Optional[bool] = None,
        cloud: typing.Optional[str] = None,
        region: typing.Union[str, collections.abc.Sequence[str], None] = None,
        enable_memory_snapshot: bool = False,
        block_network: bool = False,
        max_inputs: typing.Optional[int] = None,
        i6pn: typing.Optional[bool] = None,
        include_source: typing.Optional[bool] = None,
        _experimental_scheduler_placement: typing.Optional[modal.scheduler_placement.SchedulerPlacement] = None,
        _experimental_buffer_containers: typing.Optional[int] = None,
        _experimental_proxy_ip: typing.Optional[str] = None,
        _experimental_custom_scaling_factor: typing.Optional[float] = None,
    ) -> _FunctionDecoratorType: ...
    @typing_extensions.dataclass_transform(
        field_specifiers=(modal.cls.parameter,),
        kw_only_default=True,
    )
    def cls(
        self,
        _warn_parentheses_missing: typing.Optional[bool] = None,
        *,
        image: typing.Optional[modal.image._Image] = None,
        secrets: collections.abc.Sequence[modal.secret._Secret] = (),
        gpu: typing.Union[
            None, bool, str, modal.gpu._GPUConfig, list[typing.Union[None, bool, str, modal.gpu._GPUConfig]]
        ] = None,
        serialized: bool = False,
        mounts: collections.abc.Sequence[modal.mount._Mount] = (),
        network_file_systems: dict[
            typing.Union[str, pathlib.PurePosixPath], modal.network_file_system._NetworkFileSystem
        ] = {},
        volumes: dict[
            typing.Union[str, pathlib.PurePosixPath],
            typing.Union[modal.volume._Volume, modal.cloud_bucket_mount._CloudBucketMount],
        ] = {},
        allow_cross_region_volumes: bool = False,
        cpu: typing.Union[float, tuple[float, float], None] = None,
        memory: typing.Union[int, tuple[int, int], None] = None,
        ephemeral_disk: typing.Optional[int] = None,
        proxy: typing.Optional[modal.proxy._Proxy] = None,
        retries: typing.Union[int, modal.retries.Retries, None] = None,
        concurrency_limit: typing.Optional[int] = None,
        allow_concurrent_inputs: typing.Optional[int] = None,
        container_idle_timeout: typing.Optional[int] = None,
        timeout: typing.Optional[int] = None,
        keep_warm: typing.Optional[int] = None,
        cloud: typing.Optional[str] = None,
        region: typing.Union[str, collections.abc.Sequence[str], None] = None,
        enable_memory_snapshot: bool = False,
        block_network: bool = False,
        max_inputs: typing.Optional[int] = None,
        include_source: typing.Optional[bool] = None,
        _experimental_scheduler_placement: typing.Optional[modal.scheduler_placement.SchedulerPlacement] = None,
        _experimental_buffer_containers: typing.Optional[int] = None,
        _experimental_proxy_ip: typing.Optional[str] = None,
        _experimental_custom_scaling_factor: typing.Optional[float] = None,
    ) -> collections.abc.Callable[[CLS_T], CLS_T]: ...
    async def spawn_sandbox(
        self,
        *entrypoint_args: str,
        image: typing.Optional[modal.image._Image] = None,
        mounts: collections.abc.Sequence[modal.mount._Mount] = (),
        secrets: collections.abc.Sequence[modal.secret._Secret] = (),
        network_file_systems: dict[
            typing.Union[str, pathlib.PurePosixPath], modal.network_file_system._NetworkFileSystem
        ] = {},
        timeout: typing.Optional[int] = None,
        workdir: typing.Optional[str] = None,
        gpu: typing.Union[None, bool, str, modal.gpu._GPUConfig] = None,
        cloud: typing.Optional[str] = None,
        region: typing.Union[str, collections.abc.Sequence[str], None] = None,
        cpu: typing.Union[float, tuple[float, float], None] = None,
        memory: typing.Union[int, tuple[int, int], None] = None,
        block_network: bool = False,
        volumes: dict[
            typing.Union[str, pathlib.PurePosixPath],
            typing.Union[modal.volume._Volume, modal.cloud_bucket_mount._CloudBucketMount],
        ] = {},
        pty_info: typing.Optional[modal_proto.api_pb2.PTYInfo] = None,
        _experimental_scheduler_placement: typing.Optional[modal.scheduler_placement.SchedulerPlacement] = None,
    ) -> None: ...
    def include(self, /, other_app: _App): ...
    def _logs(
        self, client: typing.Optional[modal.client._Client] = None
    ) -> collections.abc.AsyncGenerator[str, None]: ...
    @classmethod
    def _get_container_app(cls) -> typing.Optional[_App]: ...
    @classmethod
    def _reset_container_app(cls): ...

SUPERSELF = typing.TypeVar("SUPERSELF", covariant=True)

class App:
    _all_apps: typing.ClassVar[dict[typing.Optional[str], list[App]]]
    _container_app: typing.ClassVar[typing.Optional[App]]
    _name: typing.Optional[str]
    _description: typing.Optional[str]
    _functions: dict[str, modal.functions.Function]
    _classes: dict[str, modal.cls.Cls]
    _image: typing.Optional[modal.image.Image]
    _mounts: collections.abc.Sequence[modal.mount.Mount]
    _secrets: collections.abc.Sequence[modal.secret.Secret]
    _volumes: dict[typing.Union[str, pathlib.PurePosixPath], modal.volume.Volume]
    _web_endpoints: list[str]
    _local_entrypoints: dict[str, LocalEntrypoint]
    _app_id: typing.Optional[str]
    _running_app: typing.Optional[modal.running_app.RunningApp]
    _client: typing.Optional[modal.client.Client]
    _include_source_default: typing.Optional[bool]

    def __init__(
        self,
        name: typing.Optional[str] = None,
        *,
        image: typing.Optional[modal.image.Image] = None,
        mounts: collections.abc.Sequence[modal.mount.Mount] = [],
        secrets: collections.abc.Sequence[modal.secret.Secret] = [],
        volumes: dict[typing.Union[str, pathlib.PurePosixPath], modal.volume.Volume] = {},
        include_source: typing.Optional[bool] = None,
    ) -> None: ...
    @property
    def name(self) -> typing.Optional[str]: ...
    @property
    def is_interactive(self) -> bool: ...
    @property
    def app_id(self) -> typing.Optional[str]: ...
    @property
    def description(self) -> typing.Optional[str]: ...

    class __lookup_spec(typing_extensions.Protocol):
        def __call__(
            self,
            name: str,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            create_if_missing: bool = False,
        ) -> App: ...
        async def aio(
            self,
            name: str,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            create_if_missing: bool = False,
        ) -> App: ...

    lookup: __lookup_spec

    def set_description(self, description: str): ...
    def _validate_blueprint_value(self, key: str, value: typing.Any): ...
    @property
    def image(self) -> modal.image.Image: ...
    @image.setter
    def image(self, value): ...
    def _uncreate_all_objects(self): ...

    class ___set_local_app_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self, client: modal.client.Client, running_app: modal.running_app.RunningApp
        ) -> synchronicity.combined_types.AsyncAndBlockingContextManager[None]: ...
        def aio(
            self, client: modal.client.Client, running_app: modal.running_app.RunningApp
        ) -> typing.AsyncContextManager[None]: ...

    _set_local_app: ___set_local_app_spec[typing_extensions.Self]

    class __run_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self,
            client: typing.Optional[modal.client.Client] = None,
            show_progress: typing.Optional[bool] = None,
            detach: bool = False,
            interactive: bool = False,
            environment_name: typing.Optional[str] = None,
        ) -> synchronicity.combined_types.AsyncAndBlockingContextManager[App]: ...
        def aio(
            self,
            client: typing.Optional[modal.client.Client] = None,
            show_progress: typing.Optional[bool] = None,
            detach: bool = False,
            interactive: bool = False,
            environment_name: typing.Optional[str] = None,
        ) -> typing.AsyncContextManager[App]: ...

    run: __run_spec[typing_extensions.Self]

    def _get_default_image(self): ...
    def _get_watch_mounts(self): ...
    def _add_function(self, function: modal.functions.Function, is_web_endpoint: bool): ...
    def _add_class(self, tag: str, cls: modal.cls.Cls): ...
    def _init_container(self, client: modal.client.Client, running_app: modal.running_app.RunningApp): ...
    @property
    def registered_functions(self) -> dict[str, modal.functions.Function]: ...
    @property
    def registered_classes(self) -> dict[str, modal.cls.Cls]: ...
    @property
    def registered_entrypoints(self) -> dict[str, LocalEntrypoint]: ...
    @property
    def indexed_objects(self) -> dict[str, modal.object.Object]: ...
    @property
    def registered_web_endpoints(self) -> list[str]: ...
    def local_entrypoint(
        self, _warn_parentheses_missing: typing.Any = None, *, name: typing.Optional[str] = None
    ) -> collections.abc.Callable[[collections.abc.Callable[..., typing.Any]], LocalEntrypoint]: ...
    def function(
        self,
        _warn_parentheses_missing: typing.Any = None,
        *,
        image: typing.Optional[modal.image.Image] = None,
        schedule: typing.Optional[modal.schedule.Schedule] = None,
        secrets: collections.abc.Sequence[modal.secret.Secret] = (),
        gpu: typing.Union[
            None, bool, str, modal.gpu._GPUConfig, list[typing.Union[None, bool, str, modal.gpu._GPUConfig]]
        ] = None,
        serialized: bool = False,
        mounts: collections.abc.Sequence[modal.mount.Mount] = (),
        network_file_systems: dict[
            typing.Union[str, pathlib.PurePosixPath], modal.network_file_system.NetworkFileSystem
        ] = {},
        volumes: dict[
            typing.Union[str, pathlib.PurePosixPath],
            typing.Union[modal.volume.Volume, modal.cloud_bucket_mount.CloudBucketMount],
        ] = {},
        allow_cross_region_volumes: bool = False,
        cpu: typing.Union[float, tuple[float, float], None] = None,
        memory: typing.Union[int, tuple[int, int], None] = None,
        ephemeral_disk: typing.Optional[int] = None,
        proxy: typing.Optional[modal.proxy.Proxy] = None,
        retries: typing.Union[int, modal.retries.Retries, None] = None,
        concurrency_limit: typing.Optional[int] = None,
        allow_concurrent_inputs: typing.Optional[int] = None,
        container_idle_timeout: typing.Optional[int] = None,
        timeout: typing.Optional[int] = None,
        keep_warm: typing.Optional[int] = None,
        name: typing.Optional[str] = None,
        is_generator: typing.Optional[bool] = None,
        cloud: typing.Optional[str] = None,
        region: typing.Union[str, collections.abc.Sequence[str], None] = None,
        enable_memory_snapshot: bool = False,
        block_network: bool = False,
        max_inputs: typing.Optional[int] = None,
        i6pn: typing.Optional[bool] = None,
        include_source: typing.Optional[bool] = None,
        _experimental_scheduler_placement: typing.Optional[modal.scheduler_placement.SchedulerPlacement] = None,
        _experimental_buffer_containers: typing.Optional[int] = None,
        _experimental_proxy_ip: typing.Optional[str] = None,
        _experimental_custom_scaling_factor: typing.Optional[float] = None,
    ) -> _FunctionDecoratorType: ...
    @typing_extensions.dataclass_transform(
        field_specifiers=(modal.cls.parameter,),
        kw_only_default=True,
    )
    def cls(
        self,
        _warn_parentheses_missing: typing.Optional[bool] = None,
        *,
        image: typing.Optional[modal.image.Image] = None,
        secrets: collections.abc.Sequence[modal.secret.Secret] = (),
        gpu: typing.Union[
            None, bool, str, modal.gpu._GPUConfig, list[typing.Union[None, bool, str, modal.gpu._GPUConfig]]
        ] = None,
        serialized: bool = False,
        mounts: collections.abc.Sequence[modal.mount.Mount] = (),
        network_file_systems: dict[
            typing.Union[str, pathlib.PurePosixPath], modal.network_file_system.NetworkFileSystem
        ] = {},
        volumes: dict[
            typing.Union[str, pathlib.PurePosixPath],
            typing.Union[modal.volume.Volume, modal.cloud_bucket_mount.CloudBucketMount],
        ] = {},
        allow_cross_region_volumes: bool = False,
        cpu: typing.Union[float, tuple[float, float], None] = None,
        memory: typing.Union[int, tuple[int, int], None] = None,
        ephemeral_disk: typing.Optional[int] = None,
        proxy: typing.Optional[modal.proxy.Proxy] = None,
        retries: typing.Union[int, modal.retries.Retries, None] = None,
        concurrency_limit: typing.Optional[int] = None,
        allow_concurrent_inputs: typing.Optional[int] = None,
        container_idle_timeout: typing.Optional[int] = None,
        timeout: typing.Optional[int] = None,
        keep_warm: typing.Optional[int] = None,
        cloud: typing.Optional[str] = None,
        region: typing.Union[str, collections.abc.Sequence[str], None] = None,
        enable_memory_snapshot: bool = False,
        block_network: bool = False,
        max_inputs: typing.Optional[int] = None,
        include_source: typing.Optional[bool] = None,
        _experimental_scheduler_placement: typing.Optional[modal.scheduler_placement.SchedulerPlacement] = None,
        _experimental_buffer_containers: typing.Optional[int] = None,
        _experimental_proxy_ip: typing.Optional[str] = None,
        _experimental_custom_scaling_factor: typing.Optional[float] = None,
    ) -> collections.abc.Callable[[CLS_T], CLS_T]: ...

    class __spawn_sandbox_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self,
            *entrypoint_args: str,
            image: typing.Optional[modal.image.Image] = None,
            mounts: collections.abc.Sequence[modal.mount.Mount] = (),
            secrets: collections.abc.Sequence[modal.secret.Secret] = (),
            network_file_systems: dict[
                typing.Union[str, pathlib.PurePosixPath], modal.network_file_system.NetworkFileSystem
            ] = {},
            timeout: typing.Optional[int] = None,
            workdir: typing.Optional[str] = None,
            gpu: typing.Union[None, bool, str, modal.gpu._GPUConfig] = None,
            cloud: typing.Optional[str] = None,
            region: typing.Union[str, collections.abc.Sequence[str], None] = None,
            cpu: typing.Union[float, tuple[float, float], None] = None,
            memory: typing.Union[int, tuple[int, int], None] = None,
            block_network: bool = False,
            volumes: dict[
                typing.Union[str, pathlib.PurePosixPath],
                typing.Union[modal.volume.Volume, modal.cloud_bucket_mount.CloudBucketMount],
            ] = {},
            pty_info: typing.Optional[modal_proto.api_pb2.PTYInfo] = None,
            _experimental_scheduler_placement: typing.Optional[modal.scheduler_placement.SchedulerPlacement] = None,
        ) -> None: ...
        async def aio(
            self,
            *entrypoint_args: str,
            image: typing.Optional[modal.image.Image] = None,
            mounts: collections.abc.Sequence[modal.mount.Mount] = (),
            secrets: collections.abc.Sequence[modal.secret.Secret] = (),
            network_file_systems: dict[
                typing.Union[str, pathlib.PurePosixPath], modal.network_file_system.NetworkFileSystem
            ] = {},
            timeout: typing.Optional[int] = None,
            workdir: typing.Optional[str] = None,
            gpu: typing.Union[None, bool, str, modal.gpu._GPUConfig] = None,
            cloud: typing.Optional[str] = None,
            region: typing.Union[str, collections.abc.Sequence[str], None] = None,
            cpu: typing.Union[float, tuple[float, float], None] = None,
            memory: typing.Union[int, tuple[int, int], None] = None,
            block_network: bool = False,
            volumes: dict[
                typing.Union[str, pathlib.PurePosixPath],
                typing.Union[modal.volume.Volume, modal.cloud_bucket_mount.CloudBucketMount],
            ] = {},
            pty_info: typing.Optional[modal_proto.api_pb2.PTYInfo] = None,
            _experimental_scheduler_placement: typing.Optional[modal.scheduler_placement.SchedulerPlacement] = None,
        ) -> None: ...

    spawn_sandbox: __spawn_sandbox_spec[typing_extensions.Self]

    def include(self, /, other_app: App): ...

    class ___logs_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self, client: typing.Optional[modal.client.Client] = None
        ) -> typing.Generator[str, None, None]: ...
        def aio(
            self, client: typing.Optional[modal.client.Client] = None
        ) -> collections.abc.AsyncGenerator[str, None]: ...

    _logs: ___logs_spec[typing_extensions.Self]

    @classmethod
    def _get_container_app(cls) -> typing.Optional[App]: ...
    @classmethod
    def _reset_container_app(cls): ...

class _Stub(_App):
    @staticmethod
    def __new__(
        cls,
        name: typing.Optional[str] = None,
        *,
        image: typing.Optional[modal.image._Image] = None,
        mounts: collections.abc.Sequence[modal.mount._Mount] = [],
        secrets: collections.abc.Sequence[modal.secret._Secret] = [],
        volumes: dict[typing.Union[str, pathlib.PurePosixPath], modal.volume._Volume] = {},
        include_source: typing.Optional[bool] = None,
    ): ...

class Stub(App):
    def __init__(
        self,
        name: typing.Optional[str] = None,
        *,
        image: typing.Optional[modal.image.Image] = None,
        mounts: collections.abc.Sequence[modal.mount.Mount] = [],
        secrets: collections.abc.Sequence[modal.secret.Secret] = [],
        volumes: dict[typing.Union[str, pathlib.PurePosixPath], modal.volume.Volume] = {},
        include_source: typing.Optional[bool] = None,
    ) -> None: ...

_default_image: modal.image._Image
