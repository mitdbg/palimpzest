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


def test_schema_add_fields():
    DogExtended = Dog.add_fields({"color": "The color of the dog"})
    assert sorted(DogExtended.field_names()) == ["breed", "color", "is_good"]
    assert DogExtended.field_map()["color"] == StringField(desc="The color of the dog")

    # Add the same field again, should be skipped
    DogExtended2 = DogExtended.add_fields({"color": "The color of the dog"})
    assert sorted(DogExtended2.field_names()) == ["breed", "color", "is_good"]
    assert DogExtended2.field_map()["color"] == StringField(desc="The color of the dog")

def test_schema_add_fields_with_existing_fields():
    DogExtended = Dog.add_fields({"breed": "The breed of the dog"})
    assert sorted(DogExtended.field_names()) == ["breed", "is_good"]
    assert DogExtended.field_map()["breed"] == StringField(desc="The breed of the dog")
