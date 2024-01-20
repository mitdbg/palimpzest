from typing import List, Set

#####################################################
# Basic Element types: these are the building blocks of all other Elements
# An Element is a document or part of a document. 
# It can be a text file, a paragraph, a figure, a title, a database, a record, an image, a plot, an equation, etc.
# An Element might get pretty precise, e.g., "A histogram that measures coulombic efficiency"
#####################################################
class Element():
    """Base class for all document elements"""
    def __init__(self, required=False, desc=None):
        self._required = required
        self._desc = desc

        # after it's populated, contents are available
        self._isPopulated = False
        self._data = None

    def __repr__(self):
        return f"{self.__class__.__name__}(desc={self._desc}, required={self._required})"

    def populate(self, dataDict):
        """Populate the Element with data. This is an abstract function."""
        raise NotImplementedError("You must implement this function in a subclass.")

    def getLogicalTree(self):
        """Return the logical tree of this Element. This is an abstract function."""
        raise NotImplementedError("You must implement this function in a subclass.")

    @property
    def data(self):
        """The data contained in this Element. This is a DataRecord. It is populated by a Processor."""
        if not self._isPopulated:
            raise Exception("This Element has not been populated yet. You must run a Processor on it first.")
        return self._data
    
    @property
    def children(self):
        """The child Elements of this Element"""
        return []
    
    @property
    def isPopulated(self):
        """Whether this Element has been populated by a Processor"""
        return self._isPopulated
    
    @property
    def desc(self):
        """A description of the Element"""
        return self._desc
    
    @property
    def required(self):
        """Whether this Element is required to be present in a document"""
        return self._required

class DataRecord:
    def __init__(self, element, **kwargs):
        self._element = element

        # Populate the DataRecord with the fields from the Element
        for k in self._element.__dict__:
            if k.startswith("_"):
                continue
            if k in kwargs:
                setattr(self, k, kwargs[k])
            else:
                setattr(self, k, None)

    def __repr__(self):
        keys = sorted(self.__dict__)
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
        return "{}({})".format(type(self).__name__, ", ".join(items))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


#############################
# Filters that can be applied to Element Collections
#############################
class Filter():
    """A filter that can be applied to an Element Collection"""
    def __init__(self, filterCondition: str, transformingFn=None):
        self.filterCondition = filterCondition
        self.transformingFn = transformingFn

    def __repr__(self):
        return f"{self.__class__.__name__}(filterCondition={self.filterCondition}, target={self.target})"
    
    def __eq__(self, __value: object) -> bool:
        return self.__dict__ == __value.__dict__
    
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

    def __repr__(self):
        return f"{self.__class__.__name__}(filterCondition={self.filterCondition}, target={self.target})"
    
    def _compiledFilter(self, target)->bool:
        """For TypeFilter, we don't need to wait for the compiler. We know how to implement it at authoring time."""
        return isinstance(target, type(self.target))



class AtomicElement(Element):
    """An Element that can only have one value. It's the Element version of a basic type"""
    def __init__(self, required=False, desc=None):
        super().__init__(required=required, desc=desc)
        self._data = None

    def populate(self, dataDict):
        """Populate the Element with data. This is an abstract function."""
        self._isPopulated = True
        self._data = DataRecord(self, dataDict=dataDict)
        return self._data
    
    def getLogicalTree(self):
        """Return the logical tree of this Element."""
        return (self.__repr__(), self, [])

class Field(AtomicElement):
    """A Field is a SingletonElement that contains a single value. It's untyped but probably usually a string."""
    def __init__(self, required=False, desc=None):
        super().__init__(required=required, desc=desc)

    def __repr__(self):
        return f"{self.__class__.__name__}(desc={self._desc}, required={self._required})"

class BytesField(Field):
    """A BytesField is a Field that is definitely an array of bytes."""
    def __init__(self, required=False, desc=None):
        super().__init__(required=required, desc=desc)

    def __repr__(self):
        return f"{self.__class__.__name__}(desc={self._desc}, required={self._required})"

class MultipartElement(Element):
    """A record that contains multiple Fields"""
    def __init__(self, required=False, desc=None):
        super().__init__(required=required, desc=desc)

    def populate(self, dataDict):
        """This populates the Element's subfields"""
        elementsForDataRecord = {}
        for k in self.__dict__:
            if k.startswith("_"):
                continue
            self.__dict__[k].populate(dataDict[k])
            elementsForDataRecord[k] = self.__dict__[k]._data

        self._data = DataRecord(self, **elementsForDataRecord)
        self._isPopulated = True
        return self._data

    def getLogicalTree(self):
        """Return the logical tree of this Element."""
        childTrees = [t.getLogicalTree() for t in self.children]
        return (self.__repr__(), self, childTrees)

    @property
    def children(self):
        """The child Elements of this Element"""
        children = []
        for k in self.__dict__:
            if k.startswith("_"):
                continue
            children.append(self.__dict__[k])
        return children

#####################################################
# An Element that can be one of multiple other Element kinds.
# For example, I might want to process Any([PDF, WordDoc, TextFile])
#####################################################
class Any(Element):
    """This represents ANY of the specified Element types. For example, you may not know if a document is a PDF or a Word document, but you know it's one of those two."""
    def __init__(self, possibleElements, required=False, desc=None):
        super().__init__(required=required, desc=desc)
        self._possibleElements = possibleElements

    @property
    def children(self):
        return self._possibleElements

#####################################################
# Element Collections: these contain multiple other Elements
#####################################################
class Collection(Element):
    """A Collection is an Element that contains multiple other Elements. It can be iterated over."""
    def __init__(self, basicElt, input=None, required=False, desc=None, filters=[]):
        super().__init__(required=required, desc=desc)
        self._basicElt = basicElt
        self._input = input
        self._filters = filters

    def populate(self, dataList):
        """This populates the Element Collection. Note that the PROCESSOR is responsible for implementing this Collection's filters"""
        self._isPopulated = True
        self._data = []
        for dataDict in dataList:
            self._data.append(self._basicElt.populate(dataDict))
        return self._data

    def getLogicalTree(self):
        """Return the logical tree of this Element."""
        childTrees = [t.getLogicalTree() for t in self.children]
        return (self.__repr__(), self, childTrees)

    @property
    def children(self):
        return [self._basicElt]

#    def __iter__(self):
#        """Abstract function that returns an iterator over all contents of the Collection that are of the specified type. Providing 'Element' means 'all contents'"""
#        
#        def filteredIterator():
#            for c in self.contents:
#                if all(f.test(c) for f in self.filters):
#                    yield c
#        return filteredIterator()

    def addFilterStr(self, filterCondition: str, targetFn=None):
        """Add a filter to the Collection. This filter will possibly restrict the items that are returned later."""
        if targetFn is None:
            targetFn = lambda x: x

        f = Filter(filterCondition, transformingFn=targetFn)
        return self.addFilter(f)
    
    def addFilter(self, f: Filter):
        """Add a filter to the Collection. This filter will possibly restrict the items that are returned later."""
        pass

#    def getAll(elt: Element):
#        "Abstract function that returns only Elements of the specified type."
#        pass

class List(Collection):
    """A List is an ordered Collection that contains multiple other Elements. It can be iterated over."""
    def __init__(self, basicElt: Element, input=None, required=False, desc=None, filters=[]):
        super().__init__(basicElt, input=input, required=required, desc=desc, filters=filters)

    def addFilter(self, f: Filter):
        """Add a filter to the Collection. This filter will possibly restrict the items that are returned later."""
        return List(self, input=self, required=self._required, desc="Filter(" + str(f.filterCondition) + ")", filters=[f])

class Set(Collection):
    """A Set is an unordered Collection that contains multiple other Elements. It can be iterated over."""
    def __init__(self, basicElt: Element, input=None, required=False, desc=None, filters=[]):
        super().__init__(basicElt, input=input, required=required, desc=desc, filters=filters)

    def addFilter(self, f: Filter):
        """Add a filter to the Collection. This filter will possibly restrict the items that are returned later."""
        return Set(self, input=self, required=self._required, desc="Filter(" + str(f.filterCondition) + ")", filters=[f])


###################################################################################
# "Core" useful Element types. These are Elements that almost everyone will need.
# File, TextFile, Image, PDF, etc.
###################################################################################
class File(MultipartElement):
    """A File is a record that comprises a filename and the contents of the file."""
    filename = Field(required=True, desc="The UNIX-style name of the file")
    contents = BytesField(required=True, desc="The contents of the file")

    def __init__(self, required=False, desc=None):
        super().__init__(required=required, desc=desc)

    def __repr__(self):
        return f"{self.__class__.__name__}(desc={self._desc}, filename={self.filename}, contents={self.contents}, required={self._required})"

class TextFile(File):
    """A text file is a File that contains only text. No binary data."""
    def __init__(self, required=False, desc=None):
        super().__init__(required=required, desc=desc)

class PDFFile(File):
    """A PDF file is a File that is a PDF. It has specialized fields, font information, etc."""
    # This class is currently very impoverished. It needs a lot more fields before it can correctly represent a PDF.

    def __init__(self, required=False, desc=None):
        super().__init__(required=required, desc=desc)

