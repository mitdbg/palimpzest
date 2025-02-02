import pytest

from palimpzest.core.data.datasources import MemorySource
from palimpzest.core.lib.schemas import Schema, StringField
from palimpzest.query.operators.logical import BaseScan
from palimpzest.query.optimizer.primitives import LogicalExpression
from palimpzest.query.optimizer.rules import BasicSubstitutionRule


@pytest.fixture
def schema():
    class SimpleSchema(Schema):
        filename = StringField(desc="The filename of the file")
        text = StringField(desc="The text of the file")
    return SimpleSchema

@pytest.fixture
def base_scan_op(schema):
    return BaseScan(
        datasource=MemorySource(vals=[1, 2, 3], dataset_id="test_dataset"),
        output_schema=schema
    )

def test_substitute_methods(base_scan_op):
    # Create a logical expression with the BaseScan operator
    logical_expr = LogicalExpression(
        operator=base_scan_op,
        input_group_ids=[],
        input_fields={},
        generated_fields={"some_id": StringField(desc="id"),  "text": StringField(desc="text")},
        depends_on_field_names=set(),
        group_id=1
    )

    # Apply the BasicSubstitutionRule
    physical_exprs = BasicSubstitutionRule.substitute(logical_expr, verbose=False)

    # Verify the substitution
    assert len(physical_exprs) == 1
    physical_expr = list(physical_exprs)[0]

    # Check that the operator was correctly converted to MarshalAndScanDataOp
    assert physical_expr.operator.__class__.__name__ == "MarshalAndScanDataOp"

    # Verify that the important properties were preserved
    assert physical_expr.operator.datasource == base_scan_op.datasource
    assert physical_expr.input_group_ids == logical_expr.input_group_ids
    assert physical_expr.input_fields == logical_expr.input_fields
    assert physical_expr.generated_fields == logical_expr.generated_fields
    assert physical_expr.depends_on_field_names == logical_expr.depends_on_field_names
    assert physical_expr.group_id == logical_expr.group_id
