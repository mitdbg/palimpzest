import json
import hashlib

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

    def __eq__(cls, other):
        def getDesc(o):
            attributes = dir(o)
            attributes = [attr for attr in attributes if not attr.startswith('__')]
            fields = [attr for attr in attributes if isinstance(getattr(o, attr), Field)]

            d = {k: hash(getattr(o, k)) for k in fields}
            d["__class__"] = o.__name__
            return json.dumps(d, sort_keys=True)
        
        return getDesc(cls) == getDesc(other)

    def __hash__(cls):
        attributes = dir(cls)
        attributes = [attr for attr in attributes if not attr.startswith('__')]
        fields = [attr for attr in attributes if isinstance(getattr(cls, attr), Field)]

        d = {k: hash(getattr(cls, k)) for k in fields}
        d["__class__"] = cls.__name__
        ordered = json.dumps(d, sort_keys=True)
        return hash(ordered.encode())

    def fields(cls):
        """Return a list of the fields in this Element"""
        attributes = dir(cls)
        attributes = [attr for attr in attributes if not attr.startswith('__')]
        fields = [attr for attr in attributes if isinstance(getattr(cls, attr), Field)]
        return fields

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

    def __str__(self):
        return f"{self.__class__.__name__}(desc={self._desc})"

class Field:
    """A Field is a SingletonElement that contains a single value. It's untyped but probably usually a string."""
    def __init__(self, desc=None, required=False):
        self._desc = desc
        self.required = required

    def __str__(self):
        return f"{self.__class__.__name__}(desc={self._desc})"
    
    def __hash__(self):
        return hash(self._desc + str(self.required) + self.__class__.__name__)
    
    def __eq__(self, other):
        return self._desc == other._desc and self.required == other.required and self.__class__ == other.__class__
    
    @property
    def desc(self):
        return self._desc
    
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
