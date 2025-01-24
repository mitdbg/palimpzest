"""This script contains tests for the refactoring of the physical operators"""

import os
import sys

from palimpzest.core.lib.fields import NumericField, StringField
from palimpzest.core.lib.schemas import Schema
from palimpzest.query.operators.physical import PhysicalOperator

sys.path.append("./tests/")
sys.path.append("./tests/refactor-tests/")

if not os.environ.get("OPENAI_API_KEY"):
    from palimpzest.utils.env_helpers import load_env

    load_env()


class SimpleSchema(Schema):
    name = StringField(desc="The name of the person")
    age = NumericField(desc="The age of the person")

class SimpleSchemaTwo(Schema):
    name = StringField(desc="The name of the person")
    age = NumericField(desc="The age of the person")
    height = NumericField(desc="The height of the person")

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
    schema1 = SimpleSchema()
    schema2 = SimpleSchemaTwo()

    op1 = PhysicalOperator(output_schema=schema1)
    op2 = PhysicalOperator(output_schema=schema1)
    op3 = PhysicalOperator(output_schema=schema2, verbose=True)

    assert op1 == op2  # Same output schema
    assert op1 == op1  # Same instance
    assert op1 == op1.copy()  # Copy should be equal
    assert op1 != op3  # Different parameters

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
