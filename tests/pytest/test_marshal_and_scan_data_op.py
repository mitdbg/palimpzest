from palimpzest.datamanager import DataDirectory
from palimpzest.operators import MarshalAndScanDataOp

import palimpzest as pz

import pytest

# TEST CLASS
class TestMarshalAndScanDataOp:
    def test_call(self, email_schema, emails_dataset):
        scanOp = MarshalAndScanDataOp(
            outputSchema=email_schema,
            datasetIdentifier=emails_dataset,
        )

        iterator_fn = scanOp
        for record, record_op_stats in iterator_fn:
            print(record.filename)
            pass
