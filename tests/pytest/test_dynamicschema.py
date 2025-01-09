"""This testing class tests whether we can run a workload by defining a schema dynamically."""

import palimpzest as pz
from palimpzest.constants import Model

data_path = "tests/pytest/data/"


def test_dynamicschema_jsonld():
    clinical_schema = pz.SchemaBuilder.from_file(data_path + "synapse_schema.jsonld", schema_type=pz.TextFile)
    assert clinical_schema is not None

def test_dynamicschema_csv():
    clinical_schema = pz.SchemaBuilder.from_file(data_path + "/synapse_schema.csv", schema_type=pz.TextFile)
    assert clinical_schema is not None

def test_dynamicschema_json():
    email_schema = pz.SchemaBuilder.from_file(data_path + "/email_schema.json")
    assert email_schema is not None
    assert issubclass(email_schema, pz.TextFile)

    dataset_id = "enron-eval-tiny"
    emails = pz.Dataset(dataset_id, schema=email_schema)
    emails = emails.filter(
        'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'
    )
    emails = emails.filter(
        "The email is not quoting from a news article or an article written by someone outside of Enron"
    )

    records, stats = pz.Execute(
        emails,
        policy=pz.MinCost(),
        available_models=[Model.GPT_4o_MINI],
        num_samples=3,
        nocache=True,
        allow_bonded_query=True,
        allow_code_synth=False,
        allow_token_reduction=False,
        allow_rag_reduction=False,
        allow_mixtures=False,
        execution_engine=pz.NoSentinelSequentialSingleThreadExecution,
    )

    for rec in records:
        print(rec.as_dict())


def test_dynamicschema_yml():
    email_schema = pz.SchemaBuilder.from_file(data_path + "/email_schema.yml")
    assert email_schema is not None
    assert issubclass(email_schema, pz.TextFile)

    dataset_id = "enron-eval-tiny"
    emails = pz.Dataset(dataset_id, schema=email_schema)
    emails = emails.filter(
        'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'
    )
    emails = emails.filter(
        "The email is not quoting from a news article or an article written by someone outside of Enron"
    )

    records, stats = pz.Execute(
        emails,
        policy=pz.MinCost(),
        available_models=[Model.GPT_4o_MINI],
        num_samples=3,
        nocache=True,
        allow_bonded_query=True,
        allow_code_synth=False,
        allow_token_reduction=False,
        allow_rag_reduction=False,
        allow_mixtures=False,
        execution_engine=pz.NoSentinelSequentialSingleThreadExecution,
    )

    for rec in records:
        print(rec.as_dict())
