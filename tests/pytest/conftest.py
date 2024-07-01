import palimpzest as pz
import pytest

# DEFINITIONS
ENRON_EVAL_TINY_TEST_DATA = "testdata/enron-eval-tiny"
ENRON_EVAL_TINY_DATASET_ID = "enron-eval-tiny"

@pytest.fixture
def email_schema():
    class Email(pz.TextFile):
        """Represents an email, which in practice is usually from a text file"""

        sender = pz.Field(desc="The email address of the sender", required=True)
        subject = pz.Field(desc="The subject of the email", required=True)
    
    return Email


@pytest.fixture(scope="class")
def emails_dataset():
    datasetIdentifier = ENRON_EVAL_TINY_DATASET_ID
    datadir = pz.DataDirectory()
    datadir.registerLocalDirectory(ENRON_EVAL_TINY_TEST_DATA, datasetIdentifier)

    return datasetIdentifier


@pytest.fixture
def enron_eval(email_schema):
    emails = pz.Dataset(ENRON_EVAL_TINY_DATASET_ID, schema=email_schema)
    emails = emails.filter(
        'The email refers to a fraudulent scheme (i.e., "Raptor", "Deathstar", "Chewco", and/or "Fat Boy")'
    )
    emails = emails.filter(
        "The email is not quoting from a news article or an article written by someone outside of Enron"
    )
    return emails

