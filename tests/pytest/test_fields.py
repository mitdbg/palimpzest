from palimpzest.corelib.fields import (
    BooleanField,
    StringField,
)


def test_field_equality():
    # test that equality depends on having the same desc and the same required state
    boolean_field1 = BooleanField(desc="The image has a dog", required=True)
    boolean_field2 = BooleanField(desc="The image has a dog", required=True)
    boolean_field3 = BooleanField(desc="The image has a cat", required=True)
    boolean_field4 = BooleanField(desc="The image has a dog")
    string_field1 = StringField(desc="The image has a dog", required=True)
    assert boolean_field1 == boolean_field2
    assert boolean_field1 != boolean_field3
    assert boolean_field1 != boolean_field4
    assert boolean_field1 != string_field1
