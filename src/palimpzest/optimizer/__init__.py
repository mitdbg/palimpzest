from .plan import *
from .primitives import *
from .rules import *
from .tasks import *


# TODO repeated function find a place to move it to
# https://stackoverflow.com/a/21563930
def classesinmodule(module):
    md = module.__dict__
    return [
        md[c]
        for c in md
        if (
            isinstance(md[c], type)
            and md[c].__module__ == module.__name__
            and issubclass(md[c], Rule)
        )
    ]


ALL_RULES = [*classesinmodule(rules)]
IMPLEMENTATION_RULES = [
    rule
    for rule in ALL_RULES
    if issubclass(rule, ImplementationRule)
    and rule
    not in [
        ImplementationRule,
        LLMConvertRule,
        TokenReducedConvertRule,
        CodeSynthesisConvertRule,
    ]
]
TRANSFORMATION_RULES = [
    rule
    for rule in ALL_RULES
    if issubclass(rule, TransformationRule) and rule not in [TransformationRule]
]

from .optimizer import *
