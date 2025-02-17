import collections.abc
import google.protobuf.message
import modal._object
import modal._resolver
import modal._utils.blob_utils
import modal.client
import modal.object
import modal_proto.api_pb2
import pathlib
import typing
import typing_extensions

def client_mount_name() -> str: ...
def python_standalone_mount_name(version: str) -> str: ...

class _MountEntry:
    def description(self) -> str: ...
    def get_files_to_upload(self) -> typing.Iterator[tuple[pathlib.Path, str]]: ...
    def watch_entry(self) -> tuple[pathlib.Path, pathlib.Path]: ...
    def top_level_paths(self) -> list[tuple[pathlib.Path, pathlib.PurePosixPath]]: ...

def _select_files(entries: list[_MountEntry]) -> list[tuple[pathlib.Path, pathlib.PurePosixPath]]: ...

class _MountFile(_MountEntry):
    local_file: pathlib.Path
    remote_path: pathlib.PurePosixPath

    def description(self) -> str: ...
    def get_files_to_upload(self): ...
    def watch_entry(self): ...
    def top_level_paths(self) -> list[tuple[pathlib.Path, pathlib.PurePosixPath]]: ...
    def __init__(self, local_file: pathlib.Path, remote_path: pathlib.PurePosixPath) -> None: ...
    def __repr__(self): ...
    def __eq__(self, other): ...

class _MountDir(_MountEntry):
    local_dir: pathlib.Path
    remote_path: pathlib.PurePosixPath
    ignore: collections.abc.Callable[[pathlib.Path], bool]
    recursive: bool

    def description(self): ...
    def get_files_to_upload(self): ...
    def watch_entry(self): ...
    def top_level_paths(self) -> list[tuple[pathlib.Path, pathlib.PurePosixPath]]: ...
    def __init__(
        self,
        local_dir: pathlib.Path,
        remote_path: pathlib.PurePosixPath,
        ignore: collections.abc.Callable[[pathlib.Path], bool],
        recursive: bool,
    ) -> None: ...
    def __repr__(self): ...
    def __eq__(self, other): ...

def module_mount_condition(module_base: pathlib.Path): ...
def module_mount_ignore_condition(module_base: pathlib.Path): ...

class _MountedPythonModule(_MountEntry):
    module_name: str
    remote_dir: typing.Union[pathlib.PurePosixPath, str]
    ignore: typing.Optional[collections.abc.Callable[[pathlib.Path], bool]]

    def description(self) -> str: ...
    def _proxy_entries(self) -> list[_MountEntry]: ...
    def get_files_to_upload(self) -> typing.Iterator[tuple[pathlib.Path, str]]: ...
    def watch_entry(self) -> tuple[pathlib.Path, pathlib.Path]: ...
    def top_level_paths(self) -> list[tuple[pathlib.Path, pathlib.PurePosixPath]]: ...
    def __init__(
        self,
        module_name: str,
        remote_dir: typing.Union[pathlib.PurePosixPath, str] = "/root",
        ignore: typing.Optional[collections.abc.Callable[[pathlib.Path], bool]] = None,
    ) -> None: ...
    def __repr__(self): ...
    def __eq__(self, other): ...

class NonLocalMountError(Exception): ...

class _Mount(modal._object._Object):
    _entries: typing.Optional[list[_MountEntry]]
    _deployment_name: typing.Optional[str]
    _namespace: typing.Optional[int]
    _environment_name: typing.Optional[str]
    _content_checksum_sha256_hex: typing.Optional[str]

    @staticmethod
    def _new(entries: list[_MountEntry] = []) -> _Mount: ...
    def _extend(self, entry: _MountEntry) -> _Mount: ...
    @property
    def entries(self): ...
    def _hydrate_metadata(self, handle_metadata: typing.Optional[google.protobuf.message.Message]): ...
    def _top_level_paths(self) -> list[tuple[pathlib.Path, pathlib.PurePosixPath]]: ...
    def is_local(self) -> bool: ...
    @staticmethod
    def _add_local_dir(
        local_path: pathlib.Path,
        remote_path: pathlib.PurePosixPath,
        ignore: collections.abc.Callable[[pathlib.Path], bool] = modal.file_pattern_matcher._NOTHING,
    ): ...
    def add_local_dir(
        self,
        local_path: typing.Union[str, pathlib.Path],
        *,
        remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None,
        condition: typing.Optional[collections.abc.Callable[[str], bool]] = None,
        recursive: bool = True,
    ) -> _Mount: ...
    @staticmethod
    def from_local_dir(
        local_path: typing.Union[str, pathlib.Path],
        *,
        remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None,
        condition: typing.Optional[collections.abc.Callable[[str], bool]] = None,
        recursive: bool = True,
    ) -> _Mount: ...
    @staticmethod
    def _from_local_dir(
        local_path: typing.Union[str, pathlib.Path],
        *,
        remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None,
        condition: typing.Optional[collections.abc.Callable[[str], bool]] = None,
        recursive: bool = True,
    ) -> _Mount: ...
    def add_local_file(
        self,
        local_path: typing.Union[str, pathlib.Path],
        remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None,
    ) -> _Mount: ...
    @staticmethod
    def from_local_file(
        local_path: typing.Union[str, pathlib.Path], remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None
    ) -> _Mount: ...
    @staticmethod
    def _from_local_file(
        local_path: typing.Union[str, pathlib.Path], remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None
    ) -> _Mount: ...
    @staticmethod
    def _description(entries: list[_MountEntry]) -> str: ...
    @staticmethod
    def _get_files(
        entries: list[_MountEntry],
    ) -> collections.abc.AsyncGenerator[modal._utils.blob_utils.FileUploadSpec, None]: ...
    async def _load_mount(
        self: _Mount, resolver: modal._resolver.Resolver, existing_object_id: typing.Optional[str]
    ): ...
    @staticmethod
    def from_local_python_packages(
        *module_names: str,
        remote_dir: typing.Union[str, pathlib.PurePosixPath] = "/root",
        condition: typing.Optional[collections.abc.Callable[[str], bool]] = None,
        ignore: typing.Union[typing.Sequence[str], collections.abc.Callable[[pathlib.Path], bool], None] = None,
    ) -> _Mount: ...
    @staticmethod
    def _from_local_python_packages(
        *module_names: str,
        remote_dir: typing.Union[str, pathlib.PurePosixPath] = "/root",
        condition: typing.Optional[collections.abc.Callable[[str], bool]] = None,
        ignore: typing.Union[typing.Sequence[str], collections.abc.Callable[[pathlib.Path], bool], None] = None,
    ) -> _Mount: ...
    @staticmethod
    def from_name(name: str, namespace=1, environment_name: typing.Optional[str] = None) -> _Mount: ...
    @classmethod
    async def lookup(
        cls: type[_Mount],
        name: str,
        namespace=1,
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
    ) -> _Mount: ...
    async def _deploy(
        self: _Mount,
        deployment_name: typing.Optional[str] = None,
        namespace=1,
        environment_name: typing.Optional[str] = None,
        client: typing.Optional[modal.client._Client] = None,
    ) -> None: ...
    def _get_metadata(self) -> modal_proto.api_pb2.MountHandleMetadata: ...

SUPERSELF = typing.TypeVar("SUPERSELF", covariant=True)

class Mount(modal.object.Object):
    _entries: typing.Optional[list[_MountEntry]]
    _deployment_name: typing.Optional[str]
    _namespace: typing.Optional[int]
    _environment_name: typing.Optional[str]
    _content_checksum_sha256_hex: typing.Optional[str]

    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def _new(entries: list[_MountEntry] = []) -> Mount: ...
    def _extend(self, entry: _MountEntry) -> Mount: ...
    @property
    def entries(self): ...
    def _hydrate_metadata(self, handle_metadata: typing.Optional[google.protobuf.message.Message]): ...
    def _top_level_paths(self) -> list[tuple[pathlib.Path, pathlib.PurePosixPath]]: ...
    def is_local(self) -> bool: ...
    @staticmethod
    def _add_local_dir(
        local_path: pathlib.Path,
        remote_path: pathlib.PurePosixPath,
        ignore: collections.abc.Callable[[pathlib.Path], bool] = modal.file_pattern_matcher._NOTHING,
    ): ...
    def add_local_dir(
        self,
        local_path: typing.Union[str, pathlib.Path],
        *,
        remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None,
        condition: typing.Optional[collections.abc.Callable[[str], bool]] = None,
        recursive: bool = True,
    ) -> Mount: ...
    @staticmethod
    def from_local_dir(
        local_path: typing.Union[str, pathlib.Path],
        *,
        remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None,
        condition: typing.Optional[collections.abc.Callable[[str], bool]] = None,
        recursive: bool = True,
    ) -> Mount: ...
    @staticmethod
    def _from_local_dir(
        local_path: typing.Union[str, pathlib.Path],
        *,
        remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None,
        condition: typing.Optional[collections.abc.Callable[[str], bool]] = None,
        recursive: bool = True,
    ) -> Mount: ...
    def add_local_file(
        self,
        local_path: typing.Union[str, pathlib.Path],
        remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None,
    ) -> Mount: ...
    @staticmethod
    def from_local_file(
        local_path: typing.Union[str, pathlib.Path], remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None
    ) -> Mount: ...
    @staticmethod
    def _from_local_file(
        local_path: typing.Union[str, pathlib.Path], remote_path: typing.Union[str, pathlib.PurePosixPath, None] = None
    ) -> Mount: ...
    @staticmethod
    def _description(entries: list[_MountEntry]) -> str: ...

    class ___get_files_spec(typing_extensions.Protocol):
        def __call__(
            self, entries: list[_MountEntry]
        ) -> typing.Generator[modal._utils.blob_utils.FileUploadSpec, None, None]: ...
        def aio(
            self, entries: list[_MountEntry]
        ) -> collections.abc.AsyncGenerator[modal._utils.blob_utils.FileUploadSpec, None]: ...

    _get_files: ___get_files_spec

    class ___load_mount_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, resolver: modal._resolver.Resolver, existing_object_id: typing.Optional[str]): ...
        async def aio(self, resolver: modal._resolver.Resolver, existing_object_id: typing.Optional[str]): ...

    _load_mount: ___load_mount_spec[typing_extensions.Self]

    @staticmethod
    def from_local_python_packages(
        *module_names: str,
        remote_dir: typing.Union[str, pathlib.PurePosixPath] = "/root",
        condition: typing.Optional[collections.abc.Callable[[str], bool]] = None,
        ignore: typing.Union[typing.Sequence[str], collections.abc.Callable[[pathlib.Path], bool], None] = None,
    ) -> Mount: ...
    @staticmethod
    def _from_local_python_packages(
        *module_names: str,
        remote_dir: typing.Union[str, pathlib.PurePosixPath] = "/root",
        condition: typing.Optional[collections.abc.Callable[[str], bool]] = None,
        ignore: typing.Union[typing.Sequence[str], collections.abc.Callable[[pathlib.Path], bool], None] = None,
    ) -> Mount: ...
    @staticmethod
    def from_name(name: str, namespace=1, environment_name: typing.Optional[str] = None) -> Mount: ...
    @classmethod
    def lookup(
        cls: type[Mount],
        name: str,
        namespace=1,
        client: typing.Optional[modal.client.Client] = None,
        environment_name: typing.Optional[str] = None,
    ) -> Mount: ...

    class ___deploy_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self,
            deployment_name: typing.Optional[str] = None,
            namespace=1,
            environment_name: typing.Optional[str] = None,
            client: typing.Optional[modal.client.Client] = None,
        ) -> None: ...
        async def aio(
            self,
            deployment_name: typing.Optional[str] = None,
            namespace=1,
            environment_name: typing.Optional[str] = None,
            client: typing.Optional[modal.client.Client] = None,
        ) -> None: ...

    _deploy: ___deploy_spec[typing_extensions.Self]

    def _get_metadata(self) -> modal_proto.api_pb2.MountHandleMetadata: ...

def _create_client_mount(): ...
def create_client_mount(): ...
def _get_client_mount(): ...
def _is_modal_path(remote_path: pathlib.PurePosixPath): ...
def get_auto_mounts() -> list[_Mount]: ...

ROOT_DIR: pathlib.PurePosixPath

PYTHON_STANDALONE_VERSIONS: dict[str, tuple[str, str]]
