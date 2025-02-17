import collections.abc
import modal._object
import modal.client
import modal.object
import modal.volume
import modal_proto.api_pb2
import pathlib
import synchronicity.combined_types
import typing
import typing_extensions

def network_file_system_mount_protos(
    validated_network_file_systems: list[tuple[str, _NetworkFileSystem]], allow_cross_region_volumes: bool
) -> list[modal_proto.api_pb2.SharedVolumeMount]: ...

class _NetworkFileSystem(modal._object._Object):
    @staticmethod
    def from_name(
        name: str, namespace=1, environment_name: typing.Optional[str] = None, create_if_missing: bool = False
    ) -> _NetworkFileSystem: ...
    @classmethod
    def ephemeral(
        cls: type[_NetworkFileSystem],
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
        _heartbeat_sleep: float = 300,
    ) -> typing.AsyncContextManager[_NetworkFileSystem]: ...
    @staticmethod
    async def lookup(
        name: str,
        namespace=1,
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
        create_if_missing: bool = False,
    ) -> _NetworkFileSystem: ...
    @staticmethod
    async def create_deployed(
        deployment_name: str,
        namespace=1,
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
    ) -> str: ...
    @staticmethod
    async def delete(
        name: str, client: typing.Optional[modal.client._Client] = None, environment_name: typing.Optional[str] = None
    ): ...
    async def write_file(
        self,
        remote_path: str,
        fp: typing.BinaryIO,
        progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
    ) -> int: ...
    def read_file(self, path: str) -> collections.abc.AsyncIterator[bytes]: ...
    def iterdir(self, path: str) -> collections.abc.AsyncIterator[modal.volume.FileEntry]: ...
    async def add_local_file(
        self,
        local_path: typing.Union[pathlib.Path, str],
        remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None,
        progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
    ): ...
    async def add_local_dir(
        self,
        local_path: typing.Union[pathlib.Path, str],
        remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None,
        progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
    ): ...
    async def listdir(self, path: str) -> list[modal.volume.FileEntry]: ...
    async def remove_file(self, path: str, recursive=False): ...

SUPERSELF = typing.TypeVar("SUPERSELF", covariant=True)

class NetworkFileSystem(modal.object.Object):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def from_name(
        name: str, namespace=1, environment_name: typing.Optional[str] = None, create_if_missing: bool = False
    ) -> NetworkFileSystem: ...
    @classmethod
    def ephemeral(
        cls: type[NetworkFileSystem],
        client: typing.Optional[modal.client.Client] = None,
        environment_name: typing.Optional[str] = None,
        _heartbeat_sleep: float = 300,
    ) -> synchronicity.combined_types.AsyncAndBlockingContextManager[NetworkFileSystem]: ...

    class __lookup_spec(typing_extensions.Protocol):
        def __call__(
            self,
            name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            create_if_missing: bool = False,
        ) -> NetworkFileSystem: ...
        async def aio(
            self,
            name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            create_if_missing: bool = False,
        ) -> NetworkFileSystem: ...

    lookup: __lookup_spec

    class __create_deployed_spec(typing_extensions.Protocol):
        def __call__(
            self,
            deployment_name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
        ) -> str: ...
        async def aio(
            self,
            deployment_name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
        ) -> str: ...

    create_deployed: __create_deployed_spec

    class __delete_spec(typing_extensions.Protocol):
        def __call__(
            self,
            name: str,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
        ): ...
        async def aio(
            self,
            name: str,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
        ): ...

    delete: __delete_spec

    class __write_file_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self,
            remote_path: str,
            fp: typing.BinaryIO,
            progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
        ) -> int: ...
        async def aio(
            self,
            remote_path: str,
            fp: typing.BinaryIO,
            progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
        ) -> int: ...

    write_file: __write_file_spec[typing_extensions.Self]

    class __read_file_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, path: str) -> typing.Iterator[bytes]: ...
        def aio(self, path: str) -> collections.abc.AsyncIterator[bytes]: ...

    read_file: __read_file_spec[typing_extensions.Self]

    class __iterdir_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, path: str) -> typing.Iterator[modal.volume.FileEntry]: ...
        def aio(self, path: str) -> collections.abc.AsyncIterator[modal.volume.FileEntry]: ...

    iterdir: __iterdir_spec[typing_extensions.Self]

    class __add_local_file_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self,
            local_path: typing.Union[pathlib.Path, str],
            remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None,
            progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
        ): ...
        async def aio(
            self,
            local_path: typing.Union[pathlib.Path, str],
            remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None,
            progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
        ): ...

    add_local_file: __add_local_file_spec[typing_extensions.Self]

    class __add_local_dir_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self,
            local_path: typing.Union[pathlib.Path, str],
            remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None,
            progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
        ): ...
        async def aio(
            self,
            local_path: typing.Union[pathlib.Path, str],
            remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None,
            progress_cb: typing.Optional[collections.abc.Callable[..., typing.Any]] = None,
        ): ...

    add_local_dir: __add_local_dir_spec[typing_extensions.Self]

    class __listdir_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, path: str) -> list[modal.volume.FileEntry]: ...
        async def aio(self, path: str) -> list[modal.volume.FileEntry]: ...

    listdir: __listdir_spec[typing_extensions.Self]

    class __remove_file_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, path: str, recursive=False): ...
        async def aio(self, path: str, recursive=False): ...

    remove_file: __remove_file_spec[typing_extensions.Self]
