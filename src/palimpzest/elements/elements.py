#####################################################
# Basic Element types: these are the building blocks of all other Elements
# An Element is a document or part of a document. 
# It can be a text file, a paragraph, a figure, a title, a database, a record, an image, a plot, an equation, etc.
# An Element might get pretty precise, e.g., "A histogram that measures coulombic efficiency"
#####################################################

class ElementMetaclass(type):
    def __str__(cls):
        # Emit a string that contains the names of all the class members
        # that are Fields
        attributes = dir(cls)
        attributes = [attr for attr in attributes if not attr.startswith('__')]
        # Now test each attribute to see if it's a Field
        fields = [attr for attr in attributes if isinstance(getattr(cls, attr), Field)]
        return f"{cls.__name__}({', '.join(fields)})"

    def jsonSchema(cls):
        """The JSON representation of the schema of this Element"""
        attributes = dir(cls)
        attributes = [attr for attr in attributes if not attr.startswith('__')]
        # Now test each attribute to see if it's a Field
        fields = [attr for attr in attributes if isinstance(getattr(cls, attr), Field)]

        schema = {"properties": {}, "required": [], "type": "object", "description": cls.__doc__}
        for k in fields:
            if k.startswith("_"):
                continue
            v = getattr(cls, k)
            if v is None:
                continue

            schema["properties"][k] = v.jsonSchema()

            if v.required:
                schema["required"].append(k)
        return schema
    
class Element(metaclass=ElementMetaclass):
    """Base class for all document elements"""
    def __init__(self, desc=None):
        self._desc = desc

        # after it's populated, contents are available
        self._isPopulated = False
        self._data = None

    def __str__(self):
        return f"{self.__class__.__name__}(desc={self._desc})"

    

class AtomicElement(Element):
    """An Element that can only have one value. It's the Element version of a basic type"""
 #   def __init__(self, desc=None):
 #       super().__init__(desc=desc)
 #       self._data = None

 #   def populate(self, dataDict):
 #       """Populate the Element with data. This is an abstract function."""
 #       self._isPopulated = True
 #       self._data = DataRecord(self, dataDict=dataDict)
 #       return self._data
    
    #def getLogicalTree(self):
    #    """Return the logical tree of this Element."""
    #    return (self.__repr__(), self, [])
    
    #def schema(self):
    #    return {"description": self._desc, 
    #            "type": "string"}

class Field:
    """A Field is a SingletonElement that contains a single value. It's untyped but probably usually a string."""
    def __init__(self, desc=None, required=False):
        self._desc = desc
        self.required = required

    def __str__(self):
        return f"{self.__class__.__name__}(desc={self._desc})"
    
    def jsonSchema(self):
        return {"description": self._desc,
                "type": "string"}

class BytesField(Field):
    """A BytesField is a Field that is definitely an array of bytes."""
    def __init__(self, desc=None, required=False):
        super().__init__(desc=desc, required=required)

    def jsonSchema(self):
        return {"description": self._desc, 
                "type": "string",
                "contentEncoding": "base64",
                "contentMediaType": "application/octet-stream"}

class MultipartElement(Element):
    """A record that contains multiple Fields"""
    def __init__(self, desc=None):
        super().__init__(desc=desc)
        self._required = set()

#    def populate(self, dataDict):
#        """This populates the Element's subfields"""
#        elementsForDataRecord = {}
#        for k in self.__dict__:
#            if k.startswith("_"):
#                continue
#            self.__dict__[k].populate(dataDict[k])
#            elementsForDataRecord[k] = self.__dict__[k]._data#
#
#        self._data = DataRecord(self, **elementsForDataRecord)
#        self._isPopulated = True
#        return self._data
#
#    def getLogicalTree(self):
#        """Return the logical tree of this Element."""
#        childTrees = [t.getLogicalTree() for t in self.children]
#        return (self.__repr__(), self, childTrees)

#    @property
#    def children(self):
#        """The child Elements of this Element"""
#        children = []
#        for k in self.__dict__:
#            if k.startswith("_"):
#                continue
#            children.append(self.__dict__[k])
#        return children

#    def setRequired(self, *args):
#        """Set the specified fields as required"""
#        for f in args:
#            self._required.add(f)


#####################################################
# An Element that can be one of multiple other Element kinds.
# For example, I might want to process Any([PDF, WordDoc, TextFile])
#####################################################
class Any(Element):
    """This represents ANY of the specified Element types. For example, you may not know if a document is a PDF or a Word document, but you know it's one of those two."""
    def __init__(self, possibleElements, desc=None):
        super().__init__(desc=desc)
        self._possibleElements = possibleElements

    @property
    def children(self):
        return self._possibleElements


#class List(Element):
#    """A List is an ordered Collection that contains multiple other Elements. It can be iterated over."""
#    def __init__(self, basicElt: Element, input=None, filters=[]):
#        if len(filters) == 0:
#            desc = "A List"
#        else:
#            desc = "A List of elements that satisfy these conditions: [" + " and ".join([f.filterCondition for f in filters] + "]")
#        super().__init__(basicElt, input=input, desc=desc, filters=filters)

#    def addFilter(self, f: Filter):
#        """Add a filter to the Collection. This filter will possibly restrict the items that are returned later."""
#        return List(self, input=self, filters=[f])
#
#    @property
#    def children(self):
#        return [self._basicElt]
