import modal._object
import modal.object
import typing

class _Proxy(modal._object._Object):
    @staticmethod
    def from_name(name: str, environment_name: typing.Optional[str] = None) -> _Proxy: ...

class Proxy(modal.object.Object):
    def __init__(self, *args, **kwargs): ...
    @staticmethod
    def from_name(name: str, environment_name: typing.Optional[str] = None) -> Proxy: ...
