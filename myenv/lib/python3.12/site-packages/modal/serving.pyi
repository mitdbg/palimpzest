import collections.abc
import multiprocessing.context
import multiprocessing.synchronize
import synchronicity.combined_types
import typing
import typing_extensions

_App = typing.TypeVar("_App")

def _run_serve(
    app_ref: str,
    existing_app_id: str,
    is_ready: multiprocessing.synchronize.Event,
    environment_name: str,
    show_progress: bool,
): ...
async def _restart_serve(
    app_ref: str, existing_app_id: str, environment_name: str, timeout: float = 5.0
) -> multiprocessing.context.SpawnProcess: ...
async def _terminate(proc: typing.Optional[multiprocessing.context.SpawnProcess], timeout: float = 5.0): ...
async def _run_watch_loop(
    app_ref: str, app_id: str, watcher: collections.abc.AsyncGenerator[set[str], None], environment_name: str
): ...
def _serve_app(
    app: _App,
    app_ref: str,
    _watcher: typing.Optional[collections.abc.AsyncGenerator[set[str], None]] = None,
    environment_name: typing.Optional[str] = None,
) -> typing.AsyncContextManager[_App]: ...
def _serve_stub(*args, **kwargs): ...

class __serve_app_spec(typing_extensions.Protocol):
    def __call__(
        self,
        app: _App,
        app_ref: str,
        _watcher: typing.Optional[typing.Generator[set[str], None, None]] = None,
        environment_name: typing.Optional[str] = None,
    ) -> synchronicity.combined_types.AsyncAndBlockingContextManager[_App]: ...
    def aio(
        self,
        app: _App,
        app_ref: str,
        _watcher: typing.Optional[collections.abc.AsyncGenerator[set[str], None]] = None,
        environment_name: typing.Optional[str] = None,
    ) -> typing.AsyncContextManager[_App]: ...

serve_app: __serve_app_spec

def serve_stub(*args, **kwargs): ...
