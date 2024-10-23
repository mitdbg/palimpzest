from .aggregate import *
from .code_synthesis_convert import *
from .convert import *
from .datasource import *
from .filter import *
from .limit import *
from .logical import *
from .physical import *
from .token_reduction_convert import *


# https://stackoverflow.com/a/21563930
def classesinmodule(module):
    md = module.__dict__
    return [
        md[c]
        for c in md
        if (isinstance(md[c], type) and md[c].__module__ == module.__name__ and not issubclass(md[c], type))
    ]


PHYSICAL_OPERATORS = (
    classesinmodule(physical)
    + classesinmodule(aggregate)
    + classesinmodule(convert)
    + classesinmodule(datasource)
    + classesinmodule(filter)
    + classesinmodule(limit)
)

LOGICAL_OPERATORS = classesinmodule(logical)
