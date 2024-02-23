from .elements import *

#############################
# Filters that can be applied to Element Collections
#############################
class Filter():
    """A filter that can be applied to an Element Collection"""

    def serialize(f):
        return {"filterCondition": f.filterCondition}
    
    def deserialize(d):
        return Filter(d["filterCondition"])

    def __init__(self, filterCondition: str):
        self.filterCondition = filterCondition

    def __str__(self):
        return "Filter(" + self.filterCondition + ")"
    
    def __hash__(self):
            # Custom hash function
        # For example, hash based on the value attribute
        return hash(self.filterCondition)

    def __eq__(self, other):
        # __eq__ should be defined for consistency with __hash__
        return isinstance(other, Filter) and self.filterCondition == other.filterCondition
   
#    def test(self, objToTest)->bool:
#        """Test whether the object matches the filter condition"""
#        return self._compiledFilter(objToTest)
#
#    def _compiledFilter(self, target)->bool:
#        """This is the compiled version of the filter condition. It will be implemented at compile time."""
#        pass

