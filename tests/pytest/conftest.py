import palimpzest as pz

import pytest

# DEFINITIONS
ENRON_EVAL_TINY_TEST_DATA = "testdata/enron-eval-tiny"

@pytest.fixture
def email_schema():
    class Email(pz.TextFile):
        """Represents an email, which in practice is usually from a text file"""

        sender = pz.Field(desc="The email address of the sender", required=True)
        subject = pz.Field(desc="The subject of the email", required=True)
    
    return Email


@pytest.fixture(scope="class")
def emails_dataset():
    datasetIdentifier = "enron-eval-tiny"
    datadir = pz.DataDirectory()
    datadir.registerLocalDirectory(ENRON_EVAL_TINY_TEST_DATA, datasetIdentifier)

    return datasetIdentifier
