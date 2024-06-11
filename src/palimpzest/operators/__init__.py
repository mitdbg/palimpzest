from .physical import *
from .logical import *
from .convert import *
from .filter import *
from .hardcoded_converts import *


# https://stackoverflow.com/a/21563930
def classesinmodule(module):
    md = module.__dict__
    return [
        md[c]
        for c in md
        if (isinstance(md[c], type) 
            and md[c].__module__ == module.__name__
            and not issubclass(md[c], type)
            )
    ]


PHYSICAL_OPERATORS = (
    classesinmodule(physical)
    + classesinmodule(convert)
    + classesinmodule(filter)
    + classesinmodule(hardcoded_converts)
)

LOGICAL_OPERATORS = classesinmodule(logical)
