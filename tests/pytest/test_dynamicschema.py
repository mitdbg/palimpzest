"""This testing class tests whether we can run a workload by defining a schema dynamically."""
from pathlib import Path

from palimpzest.constants import Model
from palimpzest.core.lib.schemas import TextFile
from palimpzest.policy import MinCost
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.schemabuilder.schema_builder import SchemaBuilder

data_path = Path("tests/pytest/data/")


def test_dynamicschema_jsonld(project_root: Path):
    asset_path = str(project_root / data_path / "synapse_schema.jsonld")
    clinical_schema = SchemaBuilder.from_file(asset_path, schema_type=TextFile)
    assert clinical_schema is not None

def test_dynamicschema_csv(project_root: Path):
    asset_path = str(project_root / data_path / "synapse_schema.csv")
    clinical_schema = SchemaBuilder.from_file(asset_path, schema_type=TextFile)
    assert clinical_schema is not None


def test_dynamicschema_json(mocker, enron_workload, enron_convert, enron_filter, project_root: Path):
    asset_path = str(project_root / data_path / "email_schema.json")
    email_schema = SchemaBuilder.from_file(asset_path, schema_type=TextFile)
    assert email_schema is not None
    for field_name in TextFile.model_fields:
        assert field_name in email_schema.model_fields, f"Field {field_name} not found in the schema"

    # mock out calls to generators used by the plans which parameterize this test
    mocker.patch.object(LLMFilter, "filter", side_effect=enron_filter)
    mocker.patch.object(LLMConvertBonded, "convert", side_effect=enron_convert)

    config = QueryProcessorConfig(
        policy=MinCost(),
        available_models=[Model.GPT_4o_MINI],
        num_samples=3,
        allow_bonded_query=True,
        allow_rag_reduction=False,
        allow_mixtures=False,
        allow_critic=False,
        allow_split_merge=False,
        execution_strategy="sequential",
        optimizer_strategy="pareto",
    )
    data_record_collection = enron_workload.run(config=config)

    for rec in data_record_collection:
        print(rec.to_dict())


def test_dynamicschema_yml(mocker, enron_workload, enron_convert, enron_filter, project_root: Path):
    asset_path = str(project_root / data_path / "email_schema.yml")
    email_schema = SchemaBuilder.from_file(asset_path, schema_type=TextFile)
    assert email_schema is not None
    for field_name in TextFile.model_fields:
        assert field_name in email_schema.model_fields, f"Field {field_name} not found in the schema"

    # mock out calls to generators used by the plans which parameterize this test
    mocker.patch.object(LLMFilter, "filter", side_effect=enron_filter)
    mocker.patch.object(LLMConvertBonded, "convert", side_effect=enron_convert)

    config = QueryProcessorConfig(
        policy=MinCost(),
        available_models=[Model.GPT_4o_MINI],
        num_samples=3,
        allow_bonded_query=True,
        allow_rag_reduction=False,
        allow_mixtures=False,
        allow_critic=False,
        allow_split_merge=False,
        execution_strategy="sequential",
        optimizer_strategy="pareto",
    )
    data_record_collection = enron_workload.run(config=config)

    for rec in data_record_collection:
        print(rec.to_dict())
