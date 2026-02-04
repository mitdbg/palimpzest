from typing import Any

import pytest
from pydantic import BaseModel, Field

from palimpzest.core.lib.schemas import (
    create_schema_from_df,
    create_schema_from_fields,
    get_schema_field_names,
    project,
    union_schemas,
)


class Dog(BaseModel):
    breed: str = Field(description="The breed of the dog")
    is_good: bool = Field(description="Whether the dog is good")

class Cat(BaseModel):
    breed: str = Field(description="The breed of the cat")
    is_good: bool = Field(description="Whether the cat is good")

def test_schema_equality():
    assert Dog == Dog
    assert Dog != Cat

def test_get_schema_field_names():
    assert get_schema_field_names(Dog) == ["breed", "is_good"]
    assert get_schema_field_names(Dog, id="dog") == ["Dog.dog.breed", "Dog.dog.is_good"]

def test_project_schema():
    projected_dog = project(Dog, ["breed"])
    assert projected_dog.__name__ == "Schema['breed']"
    assert get_schema_field_names(projected_dog) == ["breed"]

    projected_dog_full = project(Dog, ["Dog.id.breed", "Dog.id.is_good", "random_field"])
    assert projected_dog_full.__name__ == "Schema['breed', 'is_good']"
    assert get_schema_field_names(projected_dog_full) == ["breed", "is_good"]

def test_create_schema_from_fields():
    fields = [
        {"name": "age", "type": int, "description": "The age of the pet"},
        {"name": "weight", "type": float, "description": "The weight of the pet"}
    ]
    pet_schema = create_schema_from_fields(fields)
    assert pet_schema.__name__ == "Schema['age', 'weight']"
    assert get_schema_field_names(pet_schema) == ["age", "weight"]
    assert pet_schema.model_fields["age"].annotation is int
    assert pet_schema.model_fields["weight"].annotation is float

def test_create_schema_from_df():
    import pandas as pd

    data = {
        "name": ["Buddy", "Mittens"],
        "age": [5, 3],
        "weight": [20.5, 10.0]
    }
    df = pd.DataFrame(data)
    pet_schema = create_schema_from_df(df)
    assert pet_schema.__name__ == "Schema['age', 'name', 'weight']"
    assert get_schema_field_names(pet_schema) == ["name", "age", "weight"]
    assert pet_schema.model_fields["name"].annotation in [str, Any]
    assert pet_schema.model_fields["age"].annotation is int
    assert pet_schema.model_fields["weight"].annotation is float

def test_union_schemas():
    unioned_schema = union_schemas([Dog, Cat])
    assert unioned_schema.__name__ == "Schema['breed', 'is_good']"
    assert get_schema_field_names(unioned_schema) == ["breed", "is_good"]
    assert unioned_schema.model_fields["breed"].annotation is str
    assert unioned_schema.model_fields["is_good"].annotation is bool

    # Test with conflicting field types
    class Fish(BaseModel):
        breed: str = Field(description="The breed of the fish")
        is_good: int = Field(description="Whether the fish is good")

    with pytest.raises(AssertionError):
        union_schemas([Dog, Fish])
