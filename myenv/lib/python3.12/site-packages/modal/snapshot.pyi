import modal._object
import modal.client
import modal.object
import typing
import typing_extensions

class _SandboxSnapshot(modal._object._Object):
    @staticmethod
    async def from_id(sandbox_snapshot_id: str, client: typing.Optional[modal.client._Client] = None): ...

class SandboxSnapshot(modal.object.Object):
    def __init__(self, *args, **kwargs): ...

    class __from_id_spec(typing_extensions.Protocol):
        def __call__(self, sandbox_snapshot_id: str, client: typing.Optional[modal.client.Client] = None): ...
        async def aio(self, sandbox_snapshot_id: str, client: typing.Optional[modal.client.Client] = None): ...

    from_id: __from_id_spec
