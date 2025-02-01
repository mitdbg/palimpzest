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


def str_to_field_type(type_str: str) -> type[Field]:
    """Convert string type name to corresponding Schema field type.
    
    Args:
        type_str: String representation of the type (e.g. 'string', 'bool', 'int')
        
    Returns:
        Corresponding Field class
        
    Raises:
        ValueError: If the type string is not recognized
    """
    type_map = {
        'string': StringField,
        'str': StringField,
        'boolean': BooleanField,
        'bool': BooleanField,
        'bytes': BytesField,
        'float': FloatField,
        'integer': IntField,
        'int': IntField,
        'list': ListField,
        'numeric': NumericField,
        'number': NumericField
    }
    
    type_str = type_str.lower()
    if type_str not in type_map:
        raise ValueError(f"Unrecognized type: {type_str}. Valid types are: {', '.join(type_map.keys())}")
    
    return type_map[type_str]