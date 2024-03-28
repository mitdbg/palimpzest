from palimpzest.elements import Schema

##
# The class for user-defined functions
#
class UserFunctionSingletonMeta(type):
    """Functions are always singletons"""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class UserFunction(metaclass=UserFunctionSingletonMeta):
    def __init__(self, udfid: str, inputSchema: Schema, outputSchema: Schema):
        self.udfid = udfid
        self.inputSchema = inputSchema
        self.outputSchema = outputSchema

    def map(self, data):
        raise Exception("Not implemented")
    
