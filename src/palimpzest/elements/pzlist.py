from palimpzest import Field

class ListField(Field, list):
    """A field representing a list of elements of specified types, with full list functionality."""
    def __init__(self, element_type, desc=None, required=False, cardinality="0..*"):
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