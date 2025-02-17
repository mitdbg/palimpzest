import modal.client
import modal.io_streams
import modal.stream_type
import typing
import typing_extensions

T = typing.TypeVar("T")

class _ContainerProcess(typing.Generic[T]):
    _process_id: typing.Optional[str]
    _stdout: modal.io_streams._StreamReader[T]
    _stderr: modal.io_streams._StreamReader[T]
    _stdin: modal.io_streams._StreamWriter
    _text: bool
    _by_line: bool
    _returncode: typing.Optional[int]

    def __init__(
        self,
        process_id: str,
        client: modal.client._Client,
        stdout: modal.stream_type.StreamType = modal.stream_type.StreamType.PIPE,
        stderr: modal.stream_type.StreamType = modal.stream_type.StreamType.PIPE,
        text: bool = True,
        by_line: bool = False,
    ) -> None: ...
    @property
    def stdout(self) -> modal.io_streams._StreamReader[T]: ...
    @property
    def stderr(self) -> modal.io_streams._StreamReader[T]: ...
    @property
    def stdin(self) -> modal.io_streams._StreamWriter: ...
    @property
    def returncode(self) -> int: ...
    async def poll(self) -> typing.Optional[int]: ...
    async def wait(self) -> int: ...
    async def attach(self, *, pty: typing.Optional[bool] = None): ...

SUPERSELF = typing.TypeVar("SUPERSELF", covariant=True)

class ContainerProcess(typing.Generic[T]):
    _process_id: typing.Optional[str]
    _stdout: modal.io_streams.StreamReader[T]
    _stderr: modal.io_streams.StreamReader[T]
    _stdin: modal.io_streams.StreamWriter
    _text: bool
    _by_line: bool
    _returncode: typing.Optional[int]

    def __init__(
        self,
        process_id: str,
        client: modal.client.Client,
        stdout: modal.stream_type.StreamType = modal.stream_type.StreamType.PIPE,
        stderr: modal.stream_type.StreamType = modal.stream_type.StreamType.PIPE,
        text: bool = True,
        by_line: bool = False,
    ) -> None: ...
    @property
    def stdout(self) -> modal.io_streams.StreamReader[T]: ...
    @property
    def stderr(self) -> modal.io_streams.StreamReader[T]: ...
    @property
    def stdin(self) -> modal.io_streams.StreamWriter: ...
    @property
    def returncode(self) -> int: ...

    class __poll_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self) -> typing.Optional[int]: ...
        async def aio(self) -> typing.Optional[int]: ...

    poll: __poll_spec[typing_extensions.Self]

    class __wait_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self) -> int: ...
        async def aio(self) -> int: ...

    wait: __wait_spec[typing_extensions.Self]

    class __attach_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, *, pty: typing.Optional[bool] = None): ...
        async def aio(self, *, pty: typing.Optional[bool] = None): ...

    attach: __attach_spec[typing_extensions.Self]
