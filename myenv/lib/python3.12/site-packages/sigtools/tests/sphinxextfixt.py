import attrs

from sigtools import modifiers, specifiers


def inner(a, b):
    raise NotImplementedError


class AClass(object):
    class_attr = True
    """class attr doc"""

    def __init__(self): # pragma: no cover
        self.abc = 123
        """instance attr doc"""

    @specifiers.forwards_to(inner)
    def outer(self, c, *args, **kwargs):
        raise NotImplementedError


@modifiers.autokwoargs
def kwo(a, b, c=1, d=2):
    raise NotImplementedError


@specifiers.forwards_to(inner)
def outer(c, *args, **kwargs):
    raise NotImplementedError


def autoforwards(d, *args, **kwargs):
    return inner(*args, **kwargs) # pragma: no cover


@attrs.define
class AttrsClass:
    one: int
    two: float = attrs.field(kw_only=True)
