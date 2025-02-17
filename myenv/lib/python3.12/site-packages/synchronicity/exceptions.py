import asyncio


class UserCodeException(Exception):
    """This is used to wrap and unwrap exceptions in "user code".

    This lets us have cleaner tracebacks without all the internal synchronicity stuff."""

    def __init__(self, exc):
        # There's always going to be one place inside synchronicity where we
        # catch the exception. We can always safely remove that frame from the
        # traceback.
        self.exc = exc


def wrap_coro_exception(coro):
    async def coro_wrapped():
        try:
            return await coro
        except StopAsyncIteration:
            raise
        except asyncio.CancelledError:
            # we don't want to wrap these since cancelled Task's are otherwise
            # not properly marked as cancelled, and then not treated correctly
            # during event loop shutdown (perhaps in other places too)
            raise
        except UserCodeException:
            raise  # Pass-through in case it got double-wrapped
        except BaseException as exc:
            exc = exc.with_traceback(exc.__traceback__.tb_next)
            raise UserCodeException(exc)

    return coro_wrapped()


async def unwrap_coro_exception(coro):
    try:
        return await coro
    except UserCodeException as uc_exc:
        uc_exc.exc.__suppress_context__ = True
        raise uc_exc.exc


class NestedEventLoops(Exception):
    pass
