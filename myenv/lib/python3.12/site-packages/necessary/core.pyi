from types import ModuleType
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from packaging.version import Version
from requirements.requirement import Requirement
from typing_extensions import TypeAlias

PackageNameType = str
PackageVersionType: TypeAlias = Union[None, str, Version]
PackageNameAndVersionType: TypeAlias = Tuple[
    PackageNameType, PackageVersionType
]
FullSpecType: TypeAlias = Union[
    PackageNameType,
    PackageNameAndVersionType,
    List[Union[PackageNameType, PackageNameAndVersionType]],
]

class NecessaryImportError(ImportError): ...

def get_module_version(
    module: ModuleType,
) -> Union[Version, None]: ...

class necessary:
    def __init__(
        self,
        modules: FullSpecType,
        soft: bool = ...,
        message: Optional[str] = ...,
        errors: Optional[Tuple[Type[Exception], ...]] = ...,
    ): ...
    def parse_modules_spec_input(
        self, modules_spec: FullSpecType
    ) -> List[Requirement]: ...
    def check_module_is_available(
        self,
        req: Requirement,
        soft_check: bool = ...,
        message: Optional[str] = ...,
    ) -> bool: ...
    def __enter__(self) -> "necessary": ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...

    if TYPE_CHECKING:  # noqa: Y002
        def __bool__(self) -> Literal[True]: ...
    else:
        def __bool__(self) -> bool: ...

_T = TypeVar("_T")

def Necessary(
    modules: FullSpecType,
    soft: bool = ...,
    message: Optional[str] = ...,
    errors: Optional[Tuple[Type[Exception], ...]] = ...,
) -> Callable[[_T], _T]: ...
