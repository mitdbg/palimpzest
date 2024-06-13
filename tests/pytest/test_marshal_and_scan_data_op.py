from palimpzest.operators import MarshalAndScanDataOp

import palimpzest as pz

import pytest

# TEST CLASS
def test_call(email_schema, emails_dataset):
    scanOp = MarshalAndScanDataOp(
        outputSchema=email_schema,
        datasetIdentifier=emails_dataset
    )

    records, record_op_stats = scanOp()
    for record in records:
        print(record.filename)
    print(record_op_stats)
