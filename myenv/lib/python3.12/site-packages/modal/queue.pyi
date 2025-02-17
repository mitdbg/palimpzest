import collections.abc
import modal._object
import modal.client
import modal.object
import synchronicity.combined_types
import typing
import typing_extensions

class _Queue(modal._object._Object):
    def __init__(self): ...
    @staticmethod
    def validate_partition_key(partition: typing.Optional[str]) -> bytes: ...
    @classmethod
    def ephemeral(
        cls: type[_Queue],
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
        _heartbeat_sleep: float = 300,
    ) -> typing.AsyncContextManager[_Queue]: ...
    @staticmethod
    def from_name(
        name: str, namespace=1, environment_name: typing.Optional[str] = None, create_if_missing: bool = False
    ) -> _Queue: ...
    @staticmethod
    async def lookup(
        name: str,
        namespace=1,
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
        create_if_missing: bool = False,
    ) -> _Queue: ...
    @staticmethod
    async def delete(
        name: str,
        *,
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
    ): ...
    async def _get_nonblocking(self, partition: typing.Optional[str], n_values: int) -> list[typing.Any]: ...
    async def _get_blocking(
        self, partition: typing.Optional[str], timeout: typing.Optional[float], n_values: int
    ) -> list[typing.Any]: ...
    async def clear(self, *, partition: typing.Optional[str] = None, all: bool = False) -> None: ...
    async def get(
        self, block: bool = True, timeout: typing.Optional[float] = None, *, partition: typing.Optional[str] = None
    ) -> typing.Optional[typing.Any]: ...
    async def get_many(
        self,
        n_values: int,
        block: bool = True,
        timeout: typing.Optional[float] = None,
        *,
        partition: typing.Optional[str] = None,
    ) -> list[typing.Any]: ...
    async def put(
        self,
        v: typing.Any,
        block: bool = True,
        timeout: typing.Optional[float] = None,
        *,
        partition: typing.Optional[str] = None,
        partition_ttl: int = 86400,
    ) -> None: ...
    async def put_many(
        self,
        vs: list[typing.Any],
        block: bool = True,
        timeout: typing.Optional[float] = None,
        *,
        partition: typing.Optional[str] = None,
        partition_ttl: int = 86400,
    ) -> None: ...
    async def _put_many_blocking(
        self,
        partition: typing.Optional[str],
        partition_ttl: int,
        vs: list[typing.Any],
        timeout: typing.Optional[float] = None,
    ): ...
    async def _put_many_nonblocking(
        self, partition: typing.Optional[str], partition_ttl: int, vs: list[typing.Any]
    ): ...
    async def len(self, *, partition: typing.Optional[str] = None, total: bool = False) -> int: ...
    def iterate(
        self, *, partition: typing.Optional[str] = None, item_poll_timeout: float = 0.0
    ) -> collections.abc.AsyncGenerator[typing.Any, None]: ...

SUPERSELF = typing.TypeVar("SUPERSELF", covariant=True)

class Queue(modal.object.Object):
    def __init__(self): ...
    @staticmethod
    def validate_partition_key(partition: typing.Optional[str]) -> bytes: ...
    @classmethod
    def ephemeral(
        cls: type[Queue],
        client: typing.Optional[modal.client.Client] = None,
        environment_name: typing.Optional[str] = None,
        _heartbeat_sleep: float = 300,
    ) -> synchronicity.combined_types.AsyncAndBlockingContextManager[Queue]: ...
    @staticmethod
    def from_name(
        name: str, namespace=1, environment_name: typing.Optional[str] = None, create_if_missing: bool = False
    ) -> Queue: ...

    class __lookup_spec(typing_extensions.Protocol):
        def __call__(
            self,
            name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            create_if_missing: bool = False,
        ) -> Queue: ...
        async def aio(
            self,
            name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            create_if_missing: bool = False,
        ) -> Queue: ...

    lookup: __lookup_spec

    class __delete_spec(typing_extensions.Protocol):
        def __call__(
            self,
            name: str,
            *,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
        ): ...
        async def aio(
            self,
            name: str,
            *,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
        ): ...

    delete: __delete_spec

    class ___get_nonblocking_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, partition: typing.Optional[str], n_values: int) -> list[typing.Any]: ...
        async def aio(self, partition: typing.Optional[str], n_values: int) -> list[typing.Any]: ...

    _get_nonblocking: ___get_nonblocking_spec[typing_extensions.Self]

    class ___get_blocking_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self, partition: typing.Optional[str], timeout: typing.Optional[float], n_values: int
        ) -> list[typing.Any]: ...
        async def aio(
            self, partition: typing.Optional[str], timeout: typing.Optional[float], n_values: int
        ) -> list[typing.Any]: ...

    _get_blocking: ___get_blocking_spec[typing_extensions.Self]

    class __clear_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, *, partition: typing.Optional[str] = None, all: bool = False) -> None: ...
        async def aio(self, *, partition: typing.Optional[str] = None, all: bool = False) -> None: ...

    clear: __clear_spec[typing_extensions.Self]

    class __get_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self, block: bool = True, timeout: typing.Optional[float] = None, *, partition: typing.Optional[str] = None
        ) -> typing.Optional[typing.Any]: ...
        async def aio(
            self, block: bool = True, timeout: typing.Optional[float] = None, *, partition: typing.Optional[str] = None
        ) -> typing.Optional[typing.Any]: ...

    get: __get_spec[typing_extensions.Self]

    class __get_many_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self,
            n_values: int,
            block: bool = True,
            timeout: typing.Optional[float] = None,
            *,
            partition: typing.Optional[str] = None,
        ) -> list[typing.Any]: ...
        async def aio(
            self,
            n_values: int,
            block: bool = True,
            timeout: typing.Optional[float] = None,
            *,
            partition: typing.Optional[str] = None,
        ) -> list[typing.Any]: ...

    get_many: __get_many_spec[typing_extensions.Self]

    class __put_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self,
            v: typing.Any,
            block: bool = True,
            timeout: typing.Optional[float] = None,
            *,
            partition: typing.Optional[str] = None,
            partition_ttl: int = 86400,
        ) -> None: ...
        async def aio(
            self,
            v: typing.Any,
            block: bool = True,
            timeout: typing.Optional[float] = None,
            *,
            partition: typing.Optional[str] = None,
            partition_ttl: int = 86400,
        ) -> None: ...

    put: __put_spec[typing_extensions.Self]

    class __put_many_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self,
            vs: list[typing.Any],
            block: bool = True,
            timeout: typing.Optional[float] = None,
            *,
            partition: typing.Optional[str] = None,
            partition_ttl: int = 86400,
        ) -> None: ...
        async def aio(
            self,
            vs: list[typing.Any],
            block: bool = True,
            timeout: typing.Optional[float] = None,
            *,
            partition: typing.Optional[str] = None,
            partition_ttl: int = 86400,
        ) -> None: ...

    put_many: __put_many_spec[typing_extensions.Self]

    class ___put_many_blocking_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self,
            partition: typing.Optional[str],
            partition_ttl: int,
            vs: list[typing.Any],
            timeout: typing.Optional[float] = None,
        ): ...
        async def aio(
            self,
            partition: typing.Optional[str],
            partition_ttl: int,
            vs: list[typing.Any],
            timeout: typing.Optional[float] = None,
        ): ...

    _put_many_blocking: ___put_many_blocking_spec[typing_extensions.Self]

    class ___put_many_nonblocking_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, partition: typing.Optional[str], partition_ttl: int, vs: list[typing.Any]): ...
        async def aio(self, partition: typing.Optional[str], partition_ttl: int, vs: list[typing.Any]): ...

    _put_many_nonblocking: ___put_many_nonblocking_spec[typing_extensions.Self]

    class __len_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, *, partition: typing.Optional[str] = None, total: bool = False) -> int: ...
        async def aio(self, *, partition: typing.Optional[str] = None, total: bool = False) -> int: ...

    len: __len_spec[typing_extensions.Self]

    class __iterate_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self, *, partition: typing.Optional[str] = None, item_poll_timeout: float = 0.0
        ) -> typing.Generator[typing.Any, None, None]: ...
        def aio(
            self, *, partition: typing.Optional[str] = None, item_poll_timeout: float = 0.0
        ) -> collections.abc.AsyncGenerator[typing.Any, None]: ...

    iterate: __iterate_spec[typing_extensions.Self]
