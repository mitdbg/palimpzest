from __future__ import annotations


class Field:
    """
    A Field is defined by its description and a boolean flag indicating if it is required (for a given Schema).
    The Field class can be subclassed to specify that values of the subclass should belong to a specific type.

    For example, if you wanted to define Fields relevant to indexing research papers, you could define a field
    representing the title of a paper, the year it was published, and the journal it was published in:

    ```python
    paper_title = Field(desc="The title of a scientific paper", required=True)
    paper_year = Field(desc="The year the paper was published", required=True)
    paper_journal = Field(desc="The name of the journal that published the paper", required=False)
    ```

    Note that because not all papers are published in journals, this field might be optional (`required=False`).
    """

    def __init__(self, desc: str = "", required: bool = False) -> None:
        self._desc = desc
        self.required = required

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(desc={self._desc})"

    def __hash__(self) -> int:
        return hash(self._desc + str(self.required) + self.__class__.__name__)

    def __eq__(self, other) -> bool:
        return self._desc == other._desc and self.required == other.required and self.__class__ == other.__class__

    @property
    def desc(self) -> str:
        return self._desc

    def json_schema(self) -> dict[str, str]:
        return {"description": self._desc, "type": "undefined"}


class BooleanField(Field):
    """A BooleanField is a Field that is True or False."""

    def __init__(self, desc: str, required: bool = False):
        super().__init__(desc=desc, required=required)

    def json_schema(self) -> dict[str, str]:
        return {"description": self._desc, "type": "boolean"}


class BytesField(Field):
    """A BytesField is a Field that is definitely an array of bytes."""

    def __init__(self, desc: str, required: bool = False):
        super().__init__(desc=desc, required=required)

    def json_schema(self) -> dict[str, str]:
        return {
            "description": self._desc,
            "type": "string",
            "contentEncoding": "base64",
            "contentMediaType": "application/octet-stream",
        }


class CallableField(Field):
    """A CallableField is a Field that stores a function."""

    def __init__(self, desc: str, required: bool = False):
        super().__init__(desc=desc, required=required)

    def json_schema(self) -> dict[str, str]:
        return {"description": self._desc, "type": "callable"}


class ListField(Field, list):
    """A field representing a list of elements of specified types, with full list functionality."""

    def __init__(self, element_type, desc: str, required=False, cardinality="0..*"):
        super().__init__(desc=desc, required=required)
        self.element_type = element_type
        self.cardinality = cardinality

    def append(self, item):
        """Append item to the list after type validation."""
        if not isinstance(item, self.element_type):
            raise TypeError(f"Item must be an instance of {self.element_type.__name__}")
        super().append(item)

    def insert(self, index, item):
        """Insert item at the specified position after type validation."""
        if not isinstance(item, self.element_type):
            raise TypeError(f"Item must be an instance of {self.element_type.__name__}")
        super().insert(index, item)

    def extend(self, iterable):
        """Extend list by appending elements from the iterable after type validation."""
        for item in iterable:
            if not isinstance(item, self.element_type):
                raise TypeError(f"All items must be instances of {self.element_type.__name__}")
        super().extend(iterable)

    def __setitem__(self, index, item):
        """Set the item at the specified index after type validation."""
        if not isinstance(item, self.element_type):
            raise TypeError(f"Item must be an instance of {self.element_type.__name__}")
        super().__setitem__(index, item)

    def __str__(self):
        return f"ListField(desc={self.desc}, items={super().__str__()})"


class NumericField(Field):
    """A NumericField is a Field that is definitely an integer or a float."""

    def __init__(self, desc: str, required: bool = False):
        super().__init__(desc=desc, required=required)

    def json_schema(self) -> dict[str, str]:
        return {"description": self._desc, "type": "numeric"}


class StringField(Field):
    """A StringField is a Field that is definitely a string of text."""

    def __init__(self, desc: str, required: bool = False):
        super().__init__(desc=desc, required=required)

    def json_schema(self) -> dict[str, str]:
        return {"description": self._desc, "type": "string"}
