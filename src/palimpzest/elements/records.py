from .elements import *

#
# DataRecord
#
class DataRecord:
    """A DataRecord is a single record of data. It has a schema that corresponds to Element"""
    def __init__(self, element):
        self._element = element

    def __setattr__(self, key, value):
        if not key.startswith("_") and not hasattr(self._element, key):
            raise Exception(f"Element {self._element} does not have a field named {key}")

        super().__setattr__(key, value)

    @property
    def element(self):
        return self._element

    def __str__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, str(self.__dict__[k])[:15]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__
