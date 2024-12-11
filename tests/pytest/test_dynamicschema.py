"""This testing class tests whether we can run a workload by defining a schema dynamically."""

import palimpzest as pz


def test_dynamicschema_jsonld():

    email_schema = pz.SchemaBuilder.from_file("data/email_schema.jsonld", schema_type=pz.TextFile)
    assert email_schema is not None
    assert issubclass(email_schema, pz.TextFile)

def test_dynamicschema_json():
    email_schema = pz.SchemaBuilder.from_file("data/email_schema.json")
    assert email_schema is not None
    assert issubclass(email_schema, pz.TextFile)

def test_dynamicschema_yml():
    email_schema = pz.SchemaBuilder.from_file("data/email_schema.yml")
    assert email_schema is not None
    assert issubclass(email_schema, pz.TextFile)

def test_dynamicschema_csv():
    email_schema = pz.SchemaBuilder.from_file("data/email_schema.csv", schema_type=pz.TextFile)
    assert email_schema is not None
    assert issubclass(email_schema, pz.TextFile)
