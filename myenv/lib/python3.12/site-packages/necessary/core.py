import importlib.metadata
import operator
import warnings
from functools import wraps
from importlib import import_module
from inspect import isclass
from types import ModuleType
from typing import List, Optional, Tuple, Type, Union

import requirements
from packaging.version import Version, parse
from requirements.requirement import Requirement

PackageNameType = str
PackageVersionType = Union[None, str, Version]
PackageNameAndVersionType = Tuple[PackageNameType, PackageVersionType]
FullSpecType = Union[
    PackageNameType,
    PackageNameAndVersionType,
    List[Union[PackageNameType, PackageNameAndVersionType]],
]


OP_TABLE = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    "~=": operator.ge,
}


def get_module_version(
    module: ModuleType,
) -> Union[Version, None]:
    """Function to get the version of a package installed on the system."""
    try:
        # package has been installed, so it has a version number
        # from pyproject.toml
        if (raw_version := getattr(module, "__version__", None)) is None:
            raw_version = importlib.metadata.version(
                module.__package__ or module.__name__
            )
        return parse(raw_version)
    except Exception as e:
        warnings.warn(
            message=f"Could not parse version of {module}. Error: {e}",
            category=UserWarning,
            stacklevel=2,
        )
        return None


class NecessaryImportError(ImportError):
    ...


class necessary:
    """Check if a module is installed and optionally check its version."""

    def __init__(
        self,
        modules: FullSpecType,
        soft: bool = False,
        message: Optional[str] = None,
        errors: Optional[Tuple[Type[Exception], ...]] = None,
    ):
        """
        Args:
            modules (FullSpecType): Either a module name, a tuple
                consisting of (module name, version), or a list containing a
                mix of the two.
            soft (bool): If True, the function will return False if the
                module is not installed. If False, the function will raise an
                ImportError.
            message (Optional[str]): If provided, the function will raise the
                given message. If your message contains "{module_name}" and
                "{module_version}", the function will replace them with the
                their respective values.
            errors (Optional[List[Type[Exception]]]): If provided, the tuple
                of errors to be caught to determine if a module is not
                installed. If not provided, default is [ModuleNotFoundError].

        Raises:
            ImportError: If the module is not installed and `soft` is False.
        """
        parsed_modules_spec = self.parse_modules_spec_input(modules)
        self._errors = errors or (ModuleNotFoundError,)
        self._necessary = all(
            self.check_module_is_available(
                req=module_spec, soft_check=soft, message=message
            )
            for module_spec in parsed_modules_spec
        )

    def parse_modules_spec_input(
        self, modules_spec: FullSpecType
    ) -> List[Requirement]:
        if not isinstance(modules_spec, list):
            modules_spec = [modules_spec]

        parsed_requirements: List[Requirement] = []

        for module_spec in modules_spec:
            if isinstance(module_spec, tuple):
                if not len(module_spec) == 2:
                    raise ValueError(
                        "When providing a tuple for `modules_spec`, it must "
                        "contain exactly two elements: (module name, version)."
                    )
                module_spec = ">=".join(map(str, module_spec))

            spec = next(requirements.parse(module_spec))
            if spec.name is None:
                raise ValueError(
                    f"Could not parse module name from {module_spec}."
                )
            parsed_requirements.append(spec)

        return parsed_requirements

    def get_error(
        self, req: Requirement, message: Optional[str] = None
    ) -> ImportError:
        spec_message = (
            " with version requirements {module_version}" if req.specs else ""
        )
        message = message or (
            "Please install module {module_name}" + spec_message + "."
        )

        module_version = ",".join("".join(s) for s in req.specs)
        module_name = req.name

        return ImportError(
            message.format(
                module_name=module_name, module_version=module_version
            )
        )

    def check_module_is_available(
        self,
        req: Requirement,
        soft_check: bool = False,
        message: Optional[str] = None,
    ) -> bool:
        # this is the message to raise in case of failure and if no custom
        # message is provided by the user.
        if message is None:
            message = "'{module_name}' is required, please install it."
            if req.specs:
                message = "version '{module_version}' of " + message

        # fist check is to see if we can import the module.
        try:
            module = import_module(str(req.name))
        except self._errors:
            if soft_check:
                return False
            raise self.get_error(req=req, message=message)

        # then let's check if a minimum version is specified and if so,
        # check it.
        module_version = get_module_version(module)
        if module_version is None:
            return True

        for op_sym, ver_str in req.specs:
            op = OP_TABLE.get(op_sym, None)
            if op is None:
                raise ValueError(f"Unknown operator {op_sym} in {req}.")
            ver = parse(ver_str)

            if not op(module_version, ver):
                if soft_check:
                    return False
                else:
                    raise self.get_error(req=req, message=message)

        return True

    def __bool__(self) -> bool:
        return self._necessary

    def __enter__(self) -> "necessary":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


def Necessary(
    modules: FullSpecType,
    soft: bool = False,
    message: Optional[str] = None,
    errors: Optional[Tuple[Type[Exception], ...]] = None,
):
    """A decorator that will raise an error when the decorated
    function or class is called and the module is not available."""

    def decorating_fn(decorated, modules=modules, soft=soft, message=message):
        to_decorate = decorated.__init__ if isclass(decorated) else decorated

        @wraps(to_decorate)
        def wrapper(*args, **kwargs):
            with necessary(
                modules,
                soft=soft,
                message=message,
                errors=errors,
            ):
                return to_decorate(*args, **kwargs)

        if isclass(decorated):
            decorated.__init__ = wrapper
            return decorated
        else:
            return wrapper

    return decorating_fn
