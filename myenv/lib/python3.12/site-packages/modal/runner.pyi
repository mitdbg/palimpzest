import modal._functions
import modal.client
import modal.cls
import modal.running_app
import modal_proto.api_pb2
import multiprocessing.synchronize
import synchronicity.combined_types
import typing
import typing_extensions

_App = typing.TypeVar("_App")

V = typing.TypeVar("V")

async def _heartbeat(client: modal.client._Client, app_id: str) -> None: ...
async def _init_local_app_existing(
    client: modal.client._Client, existing_app_id: str, environment_name: str
) -> modal.running_app.RunningApp: ...
async def _init_local_app_new(
    client: modal.client._Client,
    description: str,
    app_state: int,
    environment_name: str = "",
    interactive: bool = False,
) -> modal.running_app.RunningApp: ...
async def _init_local_app_from_name(
    client: modal.client._Client, name: str, namespace: typing.Any, environment_name: str = ""
) -> modal.running_app.RunningApp: ...
async def _create_all_objects(
    client: modal.client._Client,
    running_app: modal.running_app.RunningApp,
    functions: dict[str, modal._functions._Function],
    classes: dict[str, modal.cls._Cls],
    environment_name: str,
) -> None: ...
async def _publish_app(
    client: modal.client._Client,
    running_app: modal.running_app.RunningApp,
    app_state: int,
    functions: dict[str, modal._functions._Function],
    classes: dict[str, modal.cls._Cls],
    name: str = "",
    tag: str = "",
) -> tuple[str, list[modal_proto.api_pb2.Warning]]: ...
async def _disconnect(client: modal.client._Client, app_id: str, reason: int, exc_str: str = "") -> None: ...
async def _status_based_disconnect(
    client: modal.client._Client, app_id: str, exc_info: typing.Optional[BaseException] = None
): ...
def _run_app(
    app: _App,
    *,
    client: typing.Optional[modal.client._Client] = None,
    detach: bool = False,
    environment_name: typing.Optional[str] = None,
    interactive: bool = False,
) -> typing.AsyncContextManager[_App]: ...
async def _serve_update(
    app: _App, existing_app_id: str, is_ready: multiprocessing.synchronize.Event, environment_name: str
) -> None: ...

class DeployResult:
    app_id: str
    app_page_url: str
    app_logs_url: str
    warnings: list[str]

    def __init__(self, app_id: str, app_page_url: str, app_logs_url: str, warnings: list[str]) -> None: ...
    def __repr__(self): ...
    def __eq__(self, other): ...
    def __setattr__(self, name, value): ...
    def __delattr__(self, name): ...
    def __hash__(self): ...

async def _deploy_app(
    app: _App,
    name: typing.Optional[str] = None,
    namespace: typing.Any = 1,
    client: typing.Optional[modal.client._Client] = None,
    environment_name: typing.Optional[str] = None,
    tag: str = "",
) -> DeployResult: ...
async def _interactive_shell(
    _app: _App, cmds: list[str], environment_name: str = "", pty: bool = True, **kwargs: typing.Any
) -> None: ...
def _run_stub(*args: typing.Any, **kwargs: typing.Any): ...
def _deploy_stub(*args: typing.Any, **kwargs: typing.Any): ...

class __run_app_spec(typing_extensions.Protocol):
    def __call__(
        self,
        app: _App,
        *,
        client: typing.Optional[modal.client.Client] = None,
        detach: bool = False,
        environment_name: typing.Optional[str] = None,
        interactive: bool = False,
    ) -> synchronicity.combined_types.AsyncAndBlockingContextManager[_App]: ...
    def aio(
        self,
        app: _App,
        *,
        client: typing.Optional[modal.client.Client] = None,
        detach: bool = False,
        environment_name: typing.Optional[str] = None,
        interactive: bool = False,
    ) -> typing.AsyncContextManager[_App]: ...

run_app: __run_app_spec

class __serve_update_spec(typing_extensions.Protocol):
    def __call__(
        self, app: _App, existing_app_id: str, is_ready: multiprocessing.synchronize.Event, environment_name: str
    ) -> None: ...
    async def aio(
        self, app: _App, existing_app_id: str, is_ready: multiprocessing.synchronize.Event, environment_name: str
    ) -> None: ...

serve_update: __serve_update_spec

class __deploy_app_spec(typing_extensions.Protocol):
    def __call__(
        self,
        app: _App,
        name: typing.Optional[str] = None,
        namespace: typing.Any = 1,
        client: typing.Optional[modal.client.Client] = None,
        environment_name: typing.Optional[str] = None,
        tag: str = "",
    ) -> DeployResult: ...
    async def aio(
        self,
        app: _App,
        name: typing.Optional[str] = None,
        namespace: typing.Any = 1,
        client: typing.Optional[modal.client.Client] = None,
        environment_name: typing.Optional[str] = None,
        tag: str = "",
    ) -> DeployResult: ...

deploy_app: __deploy_app_spec

class __interactive_shell_spec(typing_extensions.Protocol):
    def __call__(
        self, _app: _App, cmds: list[str], environment_name: str = "", pty: bool = True, **kwargs: typing.Any
    ) -> None: ...
    async def aio(
        self, _app: _App, cmds: list[str], environment_name: str = "", pty: bool = True, **kwargs: typing.Any
    ) -> None: ...

interactive_shell: __interactive_shell_spec

def run_stub(*args: typing.Any, **kwargs: typing.Any): ...
def deploy_stub(*args: typing.Any, **kwargs: typing.Any): ...
