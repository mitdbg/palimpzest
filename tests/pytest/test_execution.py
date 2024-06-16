from conftest import ENRON_EVAL_TINY_DATASET_ID

from palimpzest.execution import Execute, SimpleExecution
from palimpzest.policy import MaxQuality

import pytest

class TestExecution:

    def test_set_source_dataset_id(self, enron_eval):
        simple_execution = SimpleExecution()
        simple_execution.set_source_dataset_id(enron_eval)
        assert simple_execution.source_dataset_id == ENRON_EVAL_TINY_DATASET_ID

    def test_legal_discovery(self, enron_eval):
        output = Execute(enron_eval, policy=MaxQuality(), num_samples=2, nocache=True)