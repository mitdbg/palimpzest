"""This testing class tests whether we can run a workload by defining a schema dynamically."""
from palimpzest.constants import Model
from palimpzest.core.lib.schemas import TextFile
from palimpzest.policy import MinCost
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.schemabuilder.schema_builder import SchemaBuilder

data_path = "tests/pytest/data/"


def test_dynamicschema_jsonld():
    clinical_schema = SchemaBuilder.from_file(data_path + "synapse_schema.jsonld", schema_type=TextFile)
    assert clinical_schema is not None

def test_dynamicschema_csv():
    clinical_schema = SchemaBuilder.from_file(data_path + "/synapse_schema.csv", schema_type=TextFile)
    assert clinical_schema is not None


def test_dynamicschema_json(mocker, enron_workload, enron_convert, enron_filter):
    email_schema = SchemaBuilder.from_file(data_path + "/email_schema.json")
    assert email_schema is not None
    assert issubclass(email_schema, TextFile)

    # mock out calls to generators used by the plans which parameterize this test
    mocker.patch.object(LLMFilter, "filter", side_effect=enron_filter)
    mocker.patch.object(LLMConvertBonded, "convert", side_effect=enron_convert)

    config = QueryProcessorConfig(
        policy=MinCost(),
        available_models=[Model.GPT_4o_MINI],
        num_samples=3,
        nocache=True,
        allow_bonded_query=True,
        allow_code_synth=False,
        allow_token_reduction=False,
        allow_rag_reduction=False,
        allow_mixtures=False,
        processing_strategy="no_sentinel",
        execution_strategy="sequential",
        optimizer_strategy="pareto",
    )
    data_record_collection = enron_workload.run(config=config)

    for rec in data_record_collection:
        print(rec.to_dict())


def test_dynamicschema_yml(mocker, enron_workload, enron_convert, enron_filter):
    email_schema = SchemaBuilder.from_file(data_path + "/email_schema.yml")
    assert email_schema is not None
    assert issubclass(email_schema, TextFile)

    # mock out calls to generators used by the plans which parameterize this test
    mocker.patch.object(LLMFilter, "filter", side_effect=enron_filter)
    mocker.patch.object(LLMConvertBonded, "convert", side_effect=enron_convert)

    config = QueryProcessorConfig(
        policy=MinCost(),
        available_models=[Model.GPT_4o_MINI],
        num_samples=3,
        nocache=True,
        allow_bonded_query=True,
        allow_code_synth=False,
        allow_token_reduction=False,
        allow_rag_reduction=False,
        allow_mixtures=False,
        processing_strategy="no_sentinel",
        execution_strategy="sequential",
        optimizer_strategy="pareto",
    )
    data_record_collection = enron_workload.run(config=config)

    for rec in data_record_collection:
        print(rec.to_dict())
