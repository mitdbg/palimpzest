from palimpzest.core.lib.fields import BooleanField, StringField
from palimpzest.core.lib.schemas import Schema


class Dog(Schema):
    breed = StringField(desc="The breed of the dog")
    is_good = BooleanField(desc="Whether the dog is good")

class Cat(Schema):
    breed = StringField(desc="The breed of the cat")
    is_good = BooleanField(desc="Whether the cat is good")

def test_schema_equality():
    assert Dog == Dog
    assert Dog != Cat
