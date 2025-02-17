import modal._object
import modal.client
import modal.object
import typing
import typing_extensions

class _Secret(modal._object._Object):
    @staticmethod
    def from_dict(env_dict: dict[str, typing.Optional[str]] = {}): ...
    @staticmethod
    def from_local_environ(env_keys: list[str]): ...
    @staticmethod
    def from_dotenv(path=None, *, filename=".env"): ...
    @staticmethod
    def from_name(
        name: str, namespace=1, environment_name: typing.Optional[str] = None, required_keys: list[str] = []
    ) -> _Secret: ...
    @staticmethod
    async def lookup(
        name: str,
        namespace=1,
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
        required_keys: list[str] = [],
    ) -> _Secret: ...
    @staticmethod
    async def create_deployed(
        deployment_name: str,
        env_dict: dict[str, str],
        namespace=1,
        client: typing.Optional[modal.client._Client] = None,
        environment_name: typing.Optional[str] = None,
        overwrite: bool = False,
    ) -> str: ...

class Secret(modal.object.Object):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def from_dict(env_dict: dict[str, typing.Optional[str]] = {}): ...
    @staticmethod
    def from_local_environ(env_keys: list[str]): ...
    @staticmethod
    def from_dotenv(path=None, *, filename=".env"): ...
    @staticmethod
    def from_name(
        name: str, namespace=1, environment_name: typing.Optional[str] = None, required_keys: list[str] = []
    ) -> Secret: ...

    class __lookup_spec(typing_extensions.Protocol):
        def __call__(
            self,
            name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            required_keys: list[str] = [],
        ) -> Secret: ...
        async def aio(
            self,
            name: str,
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            required_keys: list[str] = [],
        ) -> Secret: ...

    lookup: __lookup_spec

    class __create_deployed_spec(typing_extensions.Protocol):
        def __call__(
            self,
            deployment_name: str,
            env_dict: dict[str, str],
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            overwrite: bool = False,
        ) -> str: ...
        async def aio(
            self,
            deployment_name: str,
            env_dict: dict[str, str],
            namespace=1,
            client: typing.Optional[modal.client.Client] = None,
            environment_name: typing.Optional[str] = None,
            overwrite: bool = False,
        ) -> str: ...

    create_deployed: __create_deployed_spec
