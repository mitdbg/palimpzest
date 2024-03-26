from palimpzest.elements import Schema
from palimpzest.tools.profiler import Profiler

import json


class DataRecord:
    """A DataRecord is a single record of data matching some Schema."""
    def __init__(self, schema: Schema, shouldProfile=False):
        self._schema = schema

        # if profiling is set to True, collect execution statistics and history of transformations
        self._shouldProfile = shouldProfile
        #if self._shouldProfile:
            # self._stats is a dictionary mapping a physical opId --> a stats dictionary
            # for the LLM operation(s) performed during that physical operation (if applicable);
            # the self._stats dictionary is updated in solver.py as the physical operation is performed;
            # some timing statistics will also be computed inside of profiler.py when the DataRecord
            # is intercepted before being passed on to the next operation.
        self._stats = {}
        self._state = {}

            # TODO: if we want to preserve entire lineage / history **within** each record, we'll
            #       need to modify solver.py (and datasources.py?) to set self._state; I don't love
            #       this approach b/c it requires updating code in many disparate places to properly
            #       manage the _state updates. The nice thing about the current way we handle the
            #       lineage / history computation is that it is entirely confined to profiler.py.
            #
            # self._state is a dictionary mapping a physical opId --> its serialized state
            # at the end of each physical operation; this dict is updated in profiler.py

    def __setattr__(self, key, value):
        if not key.startswith("_") and not hasattr(self._schema, key):
            raise Exception(f"Schema {self._schema} does not have a field named {key}")

        super().__setattr__(key, value)

    @property
    def schema(self):
        return self._schema

    def asTextJSON(self, serialize: bool=False):
        """Return a JSON representation of this DataRecord"""
        keys = sorted(self.__dict__)
        # Make a dictionary out of the key/value pairs
        d = {k: str(self.__dict__[k]) for k in keys if not k.startswith("_") and not isinstance(self.__dict__[k] , bytes)}
        d["data type"] = str(self._schema.__name__)
        d["data type description"]  = str(self._schema.__doc__)
        if serialize and self._shouldProfile:
            return d

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
