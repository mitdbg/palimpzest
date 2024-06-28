from .strategy import *
from .model_selection import *
from .bonded_query import *
from .code_synthesis import *
from .token_reduction import *

# TODO repeated function find a place to move it to
# https://stackoverflow.com/a/21563930
def classesinmodule(module):
    md = module.__dict__
    return [
        md[c]
        for c in md
        if (isinstance(md[c], type) 
            and md[c].__module__ == module.__name__
            and issubclass(md[c], PhysicalOpStrategy)
            )
    ]


REGISTERED_STRATEGIES = [*classesinmodule(model_selection),
                        *classesinmodule(bonded_query),
                        *classesinmodule(code_synthesis)]

