import collections.abc
import google.protobuf.message
import modal._resolver
import modal.client
import typing
import typing_extensions

SUPERSELF = typing.TypeVar("SUPERSELF", covariant=True)

class Object:
    _type_prefix: typing.ClassVar[typing.Optional[str]]
    _prefix_to_type: typing.ClassVar[dict[str, type]]
    _load: typing.Optional[
        collections.abc.Callable[
            [typing_extensions.Self, modal._resolver.Resolver, typing.Optional[str]], collections.abc.Awaitable[None]
        ]
    ]
    _preload: typing.Optional[
        collections.abc.Callable[
            [typing_extensions.Self, modal._resolver.Resolver, typing.Optional[str]], collections.abc.Awaitable[None]
        ]
    ]
    _rep: str
    _is_another_app: bool
    _hydrate_lazily: bool
    _deps: typing.Optional[collections.abc.Callable[..., collections.abc.Sequence[Object]]]
    _deduplication_key: typing.Optional[
        collections.abc.Callable[[], collections.abc.Awaitable[collections.abc.Hashable]]
    ]
    _object_id: typing.Optional[str]
    _client: typing.Optional[modal.client.Client]
    _is_hydrated: bool
    _is_rehydrated: bool

    def __init__(self, *args, **kwargs): ...
    @classmethod
    def __init_subclass__(cls, type_prefix: typing.Optional[str] = None): ...

    class ___init_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(
            self,
            rep: str,
            load: typing.Optional[
                collections.abc.Callable[[SUPERSELF, modal._resolver.Resolver, typing.Optional[str]], None]
            ] = None,
            is_another_app: bool = False,
            preload: typing.Optional[
                collections.abc.Callable[[SUPERSELF, modal._resolver.Resolver, typing.Optional[str]], None]
            ] = None,
            hydrate_lazily: bool = False,
            deps: typing.Optional[collections.abc.Callable[..., collections.abc.Sequence[Object]]] = None,
            deduplication_key: typing.Optional[collections.abc.Callable[[], collections.abc.Hashable]] = None,
        ): ...
        def aio(
            self,
            rep: str,
            load: typing.Optional[
                collections.abc.Callable[
                    [SUPERSELF, modal._resolver.Resolver, typing.Optional[str]], collections.abc.Awaitable[None]
                ]
            ] = None,
            is_another_app: bool = False,
            preload: typing.Optional[
                collections.abc.Callable[
                    [SUPERSELF, modal._resolver.Resolver, typing.Optional[str]], collections.abc.Awaitable[None]
                ]
            ] = None,
            hydrate_lazily: bool = False,
            deps: typing.Optional[collections.abc.Callable[..., collections.abc.Sequence[Object]]] = None,
            deduplication_key: typing.Optional[
                collections.abc.Callable[[], collections.abc.Awaitable[collections.abc.Hashable]]
            ] = None,
        ): ...

    _init: ___init_spec[typing_extensions.Self]

    def _unhydrate(self): ...
    def _initialize_from_empty(self): ...
    def _initialize_from_other(self, other): ...
    def _hydrate(
        self, object_id: str, client: modal.client.Client, metadata: typing.Optional[google.protobuf.message.Message]
    ): ...
    def _hydrate_metadata(self, metadata: typing.Optional[google.protobuf.message.Message]): ...
    def _get_metadata(self) -> typing.Optional[google.protobuf.message.Message]: ...
    def _validate_is_hydrated(self): ...
    def clone(self) -> typing_extensions.Self: ...
    @classmethod
    def _from_loader(
        cls,
        load: collections.abc.Callable[[typing_extensions.Self, modal._resolver.Resolver, typing.Optional[str]], None],
        rep: str,
        is_another_app: bool = False,
        preload: typing.Optional[
            collections.abc.Callable[[typing_extensions.Self, modal._resolver.Resolver, typing.Optional[str]], None]
        ] = None,
        hydrate_lazily: bool = False,
        deps: typing.Optional[collections.abc.Callable[..., collections.abc.Sequence[Object]]] = None,
        deduplication_key: typing.Optional[collections.abc.Callable[[], collections.abc.Hashable]] = None,
    ): ...
    @staticmethod
    def _get_type_from_id(object_id: str) -> type[Object]: ...
    @classmethod
    def _is_id_type(cls, object_id) -> bool: ...
    @classmethod
    def _new_hydrated(
        cls,
        object_id: str,
        client: modal.client.Client,
        handle_metadata: typing.Optional[google.protobuf.message.Message],
        is_another_app: bool = False,
    ) -> typing_extensions.Self: ...
    def _hydrate_from_other(self, other: typing_extensions.Self): ...
    def __repr__(self): ...
    @property
    def local_uuid(self): ...
    @property
    def object_id(self) -> str: ...
    @property
    def client(self) -> modal.client.Client: ...
    @property
    def is_hydrated(self) -> bool: ...
    @property
    def deps(self) -> collections.abc.Callable[..., collections.abc.Sequence[Object]]: ...

    class __resolve_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, client: typing.Optional[modal.client.Client] = None): ...
        async def aio(self, client: typing.Optional[modal.client.Client] = None): ...

    resolve: __resolve_spec[typing_extensions.Self]

    class __hydrate_spec(typing_extensions.Protocol[SUPERSELF]):
        def __call__(self, client: typing.Optional[modal.client.Client] = None) -> SUPERSELF: ...
        async def aio(self, client: typing.Optional[modal.client.Client] = None) -> SUPERSELF: ...

    hydrate: __hydrate_spec[typing_extensions.Self]
