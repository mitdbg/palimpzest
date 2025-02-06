import types

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


def assert_valid_field_type(field_type: type | types.UnionType | types.GenericAlias | Field) -> str:
    """
    Assert that the field is a valid field type. Return "pz_type" if field_type is a PZ type
    and "python_type" if it is a Python type.
    """
    try:
        assert issubclass(field_type, Field), "type must be a Python type or palimpzest.core.lib.fields.Field"
        return "pz_type"
    except Exception:
        assert isinstance(field_type, (type, types.UnionType, types.GenericAlias)), "type must be a Python type or palimpzest.core.lib.fields.Field"
    
    return "python_type"


def construct_field_type(field_type: type | types.UnionType | types.GenericAlias | Field, desc: str) -> Field:
    """Convert a field type and description to the corresponding PZ field.

    Args:
        type: type for the field (e.g. str, bool, list[int], StringField, etc.)
        desc: description used in the field constructor

    Returns:
        Corresponding Field class

    Raises:
        ValueError: If the type is not recognized
    """
    # if field_type is a PZ type, construct and return the field
    if assert_valid_field_type(field_type) == "pz_type":
        return field_type(desc=desc)

    # otherwise, map the Python type to a PZ type and construct the field
    supported_types_map = {
        str: StringField,
        bool: BooleanField,
        int: IntField,
        float: FloatField,
        int | float: NumericField,
        bytes: BytesField,
        list[str]: ListField(StringField),
        list[bool]: ListField(BooleanField),
        list[int]: ListField(IntField),
        list[float]: ListField(FloatField),
        list[int | float]: ListField(NumericField),
        list[bytes]: ListField(BytesField),
    }

    if field_type not in supported_types_map:
        raise ValueError(f"Unsupported type: {field_type}. Supported types are: {list(supported_types_map.keys())}")

    # get the field class and (if applicable) element field class
    field_cls = supported_types_map[field_type]

    # construct and return the field
    return field_cls(desc=desc)
