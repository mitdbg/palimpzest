"""This script contains tests for the PhysicalOperator class."""

import os

from pydantic import BaseModel, Field

from palimpzest.query.operators.physical import PhysicalOperator

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils.env_helpers import load_env

    load_env()


class SimpleSchema(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")

class SimpleSchemaTwo(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    height: int | float = Field(description="The height of the person in cm")

def test_physical_operator_init():
    """Test basic initialization of PhysicalOperator"""

    op = PhysicalOperator(
        output_schema=SimpleSchema,
        input_schema=SimpleSchema,
        depends_on=["op1", "op2"],
        logical_op_id="logical1",
        verbose=True
    )

    assert op.output_schema == SimpleSchema
    assert op.input_schema == SimpleSchema
    assert op.depends_on == ["op1", "op2"]
    assert op.logical_op_id == "logical1"
    assert op.verbose is True

def test_physical_operator_equality():
    """Test equality comparison between PhysicalOperators"""
    op1 = PhysicalOperator(logical_op_id="abc", output_schema=SimpleSchema)
    op2 = PhysicalOperator(logical_op_id="abc", output_schema=SimpleSchema)
    op3 = PhysicalOperator(logical_op_id="def", output_schema=SimpleSchemaTwo)

    assert op1 == op2
    assert op1 == op1
    assert op1 == op1.copy()
    assert op2 != op3

def test_physical_operator_str():
    """Test string representation of PhysicalOperator"""

    op = PhysicalOperator(
        output_schema=SimpleSchema,
        input_schema=SimpleSchema
    )

    str_rep = str(op)
    assert "SimpleSchema -> PhysicalOperator -> SimpleSchema" in str_rep
    assert "age, name" in str_rep

def test_physical_operator_id_generation():
    """Test operator ID generation and hashing"""
    op = PhysicalOperator(output_schema=SimpleSchema)

    # Test that op_id is initially None
    assert op.op_id is None

    # Get op_id and verify it's generated
    op_id = op.get_op_id()
    assert op_id is not None
    assert isinstance(op_id, str)

    # Test that subsequent calls return the same id
    assert op.get_op_id() == op_id

    # Test that hash is based on op_id
    assert hash(op) == int(op_id, 16)

def test_physical_operator_copy():
    """Test copying of PhysicalOperator"""
    original = PhysicalOperator(
        output_schema=SimpleSchema,
        input_schema=SimpleSchema,
        depends_on=["op1"],
        logical_op_id="logical1",
        verbose=True
    )

    copied = original.copy()

    assert copied is not original  # Different instances
    assert copied == original  # But equal in content
    assert copied.get_op_id() == original.get_op_id()  # Same op_id
    assert copied.depends_on == original.depends_on
    assert copied.logical_op_id == original.logical_op_id
    assert copied.verbose == original.verbose
