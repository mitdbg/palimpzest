from .elements import *

#############################
# Filters that can be applied to Element Collections
#############################
class Filter():
    """A filter that can be applied to an Element Collection"""
    def __init__(self, filterCondition: str, transformingFn=None):
        self.filterCondition = filterCondition
        self.transformingFn = transformingFn

    def __str__(self):
        return "Filter(" + self.filterCondition + ")"

    def test(self, objToTest)->bool:
        """Test whether the object matches the filter condition"""
        if self.transformingFn is None:
            return self._compiledFilter(objToTest)
        else:
            return self._compiledFilter(self.transformingFn(objToTest))

    def _compiledFilter(self, target)->bool:
        """This is the compiled version of the filter condition. It will be implemented at compile time."""
        pass


class TypeFilter(Filter):
    """A filter that can be applied to an Element Collection. This filter tests whether the target is an instance of the specified type."""
    def __init__(self, target, filterCondition: str=None, targetFn=None):
        super().__init__(filterCondition=filterCondition, targetFn=targetFn)
        self.target = target

    def __str__(self):
        return f"{self.__class__.__name__}(filterCondition={self.filterCondition}, target={self.target})"
    
    def _compiledFilter(self, target)->bool:
        """For TypeFilter, we don't need to wait for the compiler. We know how to implement it at authoring time."""
        return isinstance(target, type(self.target))

