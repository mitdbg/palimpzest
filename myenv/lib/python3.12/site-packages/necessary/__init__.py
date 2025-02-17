import importlib.metadata

from .core import Necessary, necessary

__all__ = ["necessary", "Necessary"]

try:
    # package has been installed, so it has a version number
    # from pyproject.toml
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    # package hasn't been installed, so set version to "dev"
    __version__ = "dev"
