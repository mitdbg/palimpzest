from palimpzest.core.lib.fields import (
    BooleanField,
    Field,
    ListField,
    StringField,
)


def test_field_equality():
    # test that equality depends on having the same desc
    boolean_field1 = BooleanField(desc="The image has a dog")
    boolean_field2 = BooleanField(desc="The image has a dog")
    boolean_field3 = BooleanField(desc="The image has a cat")
    string_field1 = StringField(desc="The image has a dog")
    assert boolean_field1 == boolean_field2
    assert boolean_field1 != boolean_field3
    assert boolean_field1 != string_field1


def test_field_metaclass():
    # test that ListField is a metaclass which produces other field types
    list_field_type = ListField(StringField)
    assert issubclass(list_field_type, Field)
    assert list_field_type.element_type == StringField
    # assert issubclass(list_field_type, ListField) # TODO: this should be true, but requires more work
