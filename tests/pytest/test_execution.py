from palimpzest.execution import Execute
from palimpzest.policy import MaxQuality

import pytest

class TestExecution:

    def test_legal_discovery(self, enron_eval):
        output = Execute(enron_eval, policy=MaxQuality(), num_samples=2, nocache=True)