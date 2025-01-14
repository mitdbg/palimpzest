from __future__ import annotations


class Field:
    """
    A Field is defined by its description and its type. The Field class is subclassed to specify
    that values of the subclass should belong to a specific type.

    For example, if you wanted to define Fields relevant to indexing research papers, you could define a field
    representing the title of a paper, the year it was published, and the journal it was published in:

    ```python
    paper_title = Field(desc="The title of a scientific paper")
    paper_year = Field(desc="The year the paper was published")
    paper_journal = Field(desc="The name of the journal that published the paper")
    ```
    """
    is_image_field = False

    def __init__(self, desc: str = "") -> None:
        self._desc = desc
        self.type = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(desc={self._desc})"

    def __hash__(self) -> int:
        return hash(self._desc + self.__class__.__name__)

    def __eq__(self, other) -> bool:
        return self._desc == other._desc and self.__class__ == other.__class__

    @property
    def desc(self) -> str:
        return self._desc

    def json_schema(self) -> dict:
        return {"description": self._desc, "type": str(self.type)}


class BooleanField(Field):
    """A BooleanField is a Field that is True or False."""

    def __init__(self, desc: str):
        super().__init__(desc=desc)
        self.type = bool


class BytesField(Field):
    """A BytesField is a Field that is definitely an array of bytes."""

    def __init__(self, desc: str):
        super().__init__(desc=desc)
        self.type = bytes

    def json_schema(self) -> dict[str, str]:
        return {
            "description": self._desc,
            "type": str(self.type),
            "contentEncoding": "base64",
            "contentMediaType": "application/octet-stream",
        }


class CallableField(Field):
    """A CallableField is a Field that stores a function."""

    def __init__(self, desc: str):
        super().__init__(desc=desc)
        self.type = type(lambda x: x)


class FloatField(Field):
    """A FloatField is a Field that is definitely an integer or a float."""

    def __init__(self, desc: str):
        super().__init__(desc=desc)
        self.type = float


class IntField(Field):
    """An IntField is a Field that is definitely an integer or a float."""

    def __init__(self, desc: str):
        super().__init__(desc=desc)
        self.type = int


class ListField(Field, list):
    """A field representing a list of elements of specified types, with full list functionality."""

    def __init__(self, element_type: Field, desc: str):
        super().__init__(desc=desc)
        self.element_type = element_type
        self.type = list

        if element_type.is_image_field:
            self.__class__.is_image_field = True


class NumericField(Field):
    """A NumericField is a Field that is definitely a number."""

    def __init__(self, desc: str):
        super().__init__(desc=desc)
        self.type = int | float


class StringField(Field):
    """A StringField is a Field that is definitely a string of text."""

    def __init__(self, desc: str):
        super().__init__(desc=desc)
        self.type = str


class ImageFilepathField(StringField):
    """An ImageFilepathField is a StringField that contains the filepath to an image."""
    is_image_field = True

    def __init__(self, desc: str):
        super().__init__(desc=desc)


class ImageURLField(StringField):
    """An ImageURLField is a StringField that contains the publicly accessible URL for an image."""
    is_image_field = True

    def __init__(self, desc: str):
        super().__init__(desc=desc)


class ImageBase64Field(BytesField):
    """An ImageBase64Field is a BytesField that contains a base64 encoded image."""
    is_image_field = True

    def __init__(self, desc: str):
        super().__init__(desc=desc)
