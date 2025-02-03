from palimpzest.core.lib.fields import (
    BooleanField,
    BytesField,
    Field,
    FloatField,
    IntField,
    ListField,
    NumericField,
    StringField,
)


def construct_field_from_python_type(type: type, desc: str) -> type[Field]:
    """Convert Python type and description to corresponding Schema field type.

    Args:
        type: Python type for the field (e.g. str, bool, list[int], etc.)
        desc: description used in the field constructor

    Returns:
        Corresponding Field class

    Raises:
        ValueError: If the type is not recognized
    """
    supported_types_map = {
        str: (StringField, None),
        bool: (BooleanField, None),
        int: (IntField, None),
        float: (FloatField, None),
        int | float: (NumericField, None),
        bytes: (BytesField, None),
        list[str]: (ListField, StringField),
        list[bool]: (ListField, BooleanField),
        list[int]: (ListField, IntField),
        list[float]: (ListField, FloatField),
        list[int | float]: (ListField, NumericField),
        list[bytes]: (ListField, BytesField),
    }

    if type not in supported_types_map:
        raise ValueError(f"Unsupported type: {type}. Supported types are: {list(supported_types_map.keys())}")

    # get the field class and (if applicable) element field class
    field_cls, element_field_cls = supported_types_map[type]

    # construct and return the field
    if field_cls == ListField:
        return field_cls(element_type=element_field_cls, desc=desc)

    return field_cls(desc=desc)
