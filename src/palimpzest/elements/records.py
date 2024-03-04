from palimpzest.elements import Schema

import json


class DataRecord:
    """A DataRecord is a single record of data matching some Schema."""
    def __init__(self, schema: Schema):
        self._schema = schema

    def __setattr__(self, key, value):
        if not key.startswith("_") and not hasattr(self._schema, key):
            raise Exception(f"Schema {self._schema} does not have a field named {key}")

        super().__setattr__(key, value)

    @property
    def schema(self):
        return self._schema

    def asTextJSON(self):
        """Return a JSON representation of this DataRecord"""
        keys = sorted(self.__dict__)
        # Make a dictionary out of the key/value pairs
        d = {k: str(self.__dict__[k]) for k in keys if not k.startswith("_") and not isinstance(self.__dict__[k] , bytes)}
        d["data type"] = str(self._schema.__name__)
        d["data type description"]  = str(self._schema.__doc__)
        return json.dumps(d, indent=2)

    def asJSON(self):
        """Return a JSON representation of this DataRecord"""
        keys = sorted(self.__dict__)
        # Make a dictionary out of the key/value pairs
        d = {k: self.__dict__[k] for k in keys if not k.startswith("_")}
        d["data type"] = str(self._schema.__name__)
        d["data type description"]  = str(self._schema.__doc__)
        return json.dumps(d, indent=2)

    def __str__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, str(self.__dict__[k])[:15]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
