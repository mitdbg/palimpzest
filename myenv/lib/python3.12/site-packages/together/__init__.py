from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Callable

from together import (
    abstract,
    client,
    constants,
    error,
    filemanager,
    resources,
    together_response,
    types,
    utils,
)
from together.version import VERSION

from together.legacy.complete import AsyncComplete, Complete, Completion
from together.legacy.embeddings import Embeddings
from together.legacy.files import Files
from together.legacy.finetune import Finetune
from together.legacy.images import Image
from together.legacy.models import Models

version = VERSION

log: str | None = None  # Set to either 'debug' or 'info', controls console logging

if TYPE_CHECKING:
    import requests
    from aiohttp import ClientSession

requestssession: "requests.Session" | Callable[[], "requests.Session"] | None = None

aiosession: ContextVar["ClientSession" | None] = ContextVar(
    "aiohttp-session", default=None
)

from together.client import AsyncClient, AsyncTogether, Client, Together


api_key: str | None = None  # To be deprecated in the next major release

# Legacy functions


__all__ = [
    "aiosession",
    "constants",
    "version",
    "Together",
    "AsyncTogether",
    "Client",
    "AsyncClient",
    "resources",
    "types",
    "abstract",
    "filemanager",
    "error",
    "together_response",
    "client",
    "utils",
    "Complete",
    "AsyncComplete",
    "Completion",
    "Embeddings",
    "Files",
    "Finetune",
    "Image",
    "Models",
]
