import collections.abc
import modal._object
import modal.client
import modal.object
import synchronicity.combined_types
import typing
import typing_extensions

def _serialize_dict(data): ...

class _Dict(modal._object._Object):
    def __init__(self, data={}): ...
    @classmethod
    def ephemeral(
        cls: type[_Dict],
        data: typing.Optional[dict] = None,
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
        _heartbeat_sleep: float = 300,
    ) -> typing.AsyncContextManager[_Dict]: ...
    @staticmethod
    def from_name(
        name: str,
        data: typing.Optional[dict] = None,
        namespace=1,
        environment_name: typing.Optional[str] = None,
        create_if_missing: bool = False,
    ) -> _Dict: ...
    @staticmethod
    async def lookup(
        name: str,
        data: typing.Optional[dict] = None,
        namespace=1,
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
        create_if_missing: bool = False,
    ) -> _Dict: ...
    @staticmethod
    async def delete(
        name: str,
        *,
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
    ): ...
    async def clear(self) -> None: ...
    async def get(self, key: typing.Any, default: typing.Optional[typing.Any] = None) -> typing.Any: ...
    async def contains(self, key: typing.Any) -> bool: ...
    async def len(self) -> int: ...
    async def __getitem__(self, key: typing.Any) -> typing.Any: ...
    async def update(self, **kwargs) -> None: ...
    async def put(self, key: typing.Any, value: typing.Any) -> None: ...
    async def __setitem__(self, key: typing.Any, value: typing.Any) -> None: ...
    async def pop(self, key: typing.Any) -> typing.Any: ...
    async def __delitem__(self, key: typing.Any) -> typing.Any: ...
    async def __contains__(self, key: typing.Any) -> bool: ...
    def keys(self) -> collections.abc.AsyncIterator[typing.Any]: ...
    def values(self) -> collections.abc.AsyncIterator[typing.Any]: ...
    def items(self) -> collections.abc.AsyncIterator[tuple[typing.Any, typing.Any]]: ...

SUPERSELF = typing.TypeVar("SUPERSELF", covariant=True)

class Dict(modal.object.Object):
    def __init__(self, data={}): ...
    @classmethod
    def ephemeral(
        cls: type[Dict],
        data: typing.Optional[dict] = None,
        client: typing.Optional[modal.client.Client] = None,
        environment_name: typing.Optional[str] = None,
        _heartbeat_sleep: float = 300,
    ) -> synchronicity.combined_types.AsyncAndBlockingContextManager[Dict]: ...
    @staticmethod
    def from_name(
        name: str,
        data: typing.Optional[dict] = None,
        namespace=1,
        environment_name: typing.Optional[str] = None,
        create_if_missing: bool = False,
    ) -> Dict: ...

    class __lookup_spec(typing_extensions.Protocol):
        def __call__(
            self,
            name: str,
            data: typing.Optional[dict] = None,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            create_if_missing: bool = False,
        ) -> Dict: ...
        async def aio(
            self,
            name: str,
            data: typing.Optional[dict] = None,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            create_if_missing: bool = False,
        ) -> Dict: ...

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

    class __clear_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self) -> None: ...
        async def aio(self) -> None: ...

    clear: __clear_spec[typing_extensions.Self]

    class __get_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, key: typing.Any, default: typing.Optional[typing.Any] = None) -> typing.Any: ...
        async def aio(self, key: typing.Any, default: typing.Optional[typing.Any] = None) -> typing.Any: ...

    get: __get_spec[typing_extensions.Self]

    class __contains_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, key: typing.Any) -> bool: ...
        async def aio(self, key: typing.Any) -> bool: ...

    contains: __contains_spec[typing_extensions.Self]

    class __len_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self) -> int: ...
        async def aio(self) -> int: ...

    len: __len_spec[typing_extensions.Self]

    class ____getitem___spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, key: typing.Any) -> typing.Any: ...
        async def aio(self, key: typing.Any) -> typing.Any: ...

    __getitem__: ____getitem___spec[typing_extensions.Self]

    class __update_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, **kwargs) -> None: ...
        async def aio(self, **kwargs) -> None: ...

    update: __update_spec[typing_extensions.Self]

    class __put_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, key: typing.Any, value: typing.Any) -> None: ...
        async def aio(self, key: typing.Any, value: typing.Any) -> None: ...

    put: __put_spec[typing_extensions.Self]

    class ____setitem___spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, key: typing.Any, value: typing.Any) -> None: ...
        async def aio(self, key: typing.Any, value: typing.Any) -> None: ...

    __setitem__: ____setitem___spec[typing_extensions.Self]

    class __pop_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, key: typing.Any) -> typing.Any: ...
        async def aio(self, key: typing.Any) -> typing.Any: ...

    pop: __pop_spec[typing_extensions.Self]

    class ____delitem___spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, key: typing.Any) -> typing.Any: ...
        async def aio(self, key: typing.Any) -> typing.Any: ...

    __delitem__: ____delitem___spec[typing_extensions.Self]

    class ____contains___spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, key: typing.Any) -> bool: ...
        async def aio(self, key: typing.Any) -> bool: ...

    __contains__: ____contains___spec[typing_extensions.Self]

    class __keys_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self) -> typing.Iterator[typing.Any]: ...
        def aio(self) -> collections.abc.AsyncIterator[typing.Any]: ...

    keys: __keys_spec[typing_extensions.Self]

    class __values_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self) -> typing.Iterator[typing.Any]: ...
        def aio(self) -> collections.abc.AsyncIterator[typing.Any]: ...

    values: __values_spec[typing_extensions.Self]

    class __items_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self) -> typing.Iterator[tuple[typing.Any, typing.Any]]: ...
        def aio(self) -> collections.abc.AsyncIterator[tuple[typing.Any, typing.Any]]: ...

    items: __items_spec[typing_extensions.Self]
