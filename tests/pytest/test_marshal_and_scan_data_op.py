from palimpzest.operators import MarshalAndScanDataOp

import palimpzest as pz

import pytest

# TEST CLASS
def test_call(email_schema, enron_eval_tiny):
    scanOp = MarshalAndScanDataOp(
        outputSchema=email_schema,
        datasetIdentifier=enron_eval_tiny
    )

    records, record_op_stats = scanOp()
    for record in records:
        print(record.filename)
    print(record_op_stats)
