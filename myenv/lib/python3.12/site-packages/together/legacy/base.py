import functools
import warnings


API_KEY_WARNING = (
    "The use of together.api_key is deprecated and will be removed in the next major release. "
    "Please set the TOGETHER_API_KEY environment variable instead."
)


def deprecated(func):  # type: ignore
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):  # type: ignore
        warnings.warn(
            f"Call to deprecated function {func.__name__}.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return new_func
