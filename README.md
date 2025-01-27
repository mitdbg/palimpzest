![pz-banner](https://palimpzest-workloads.s3.us-east-1.amazonaws.com/palimpzest-cropped.png)

# Palimpzest (PZ)
[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv)](https://arxiv.org/pdf/2405.14696)
[![Blog Post](https://img.shields.io/badge/Blog-PZ-green)](https://dsg.csail.mit.edu/projects/palimpzest/)
[![Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zqOxnh_G6eZ8_xax6PvDr-EjMt7hp4R5?usp=sharing)
[![Video](https://img.shields.io/badge/YouTube-Talk-red?logo=youtube)](https://youtu.be/T8VQfyBiki0?si=eiph57DSEkDNbEIu)
[![PyPI](https://img.shields.io/pypi/v/palimpzest)](https://pypi.org/project/palimpzest/)
[![PyPI - Monthly Downloads](https://img.shields.io/pypi/dm/palimpzest)](https://pypi.org/project/palimpzest/)

## Getting started
You can find a stable version of the PZ package on PyPI [here](https://pypi.org/project/palimpzest/). To install the package, run:
```bash
$ pip install palimpzest
```

Alternatively, to install the latest version of the package from this repository, you can clone this repository and run the following commands:
```bash
$ git clone git@github.com:mitdbg/palimpzest.git
$ cd palimpzest
$ pip install .
```

## Quick Start
The easiest way to get started with Palimpzest is to run the `quickstart.ipynb` jupyter notebook. We demonstrate the full workflow of working with PZ, including registering a dataset, composing and executing a pipeline, and accessing the results.
To run the notebook, you can use the following command:
```bash
$ jupyter notebook
```
And then access the notebook from the jupyter interface in your browser at `localhost:8888`.

### Even Quicker Start
For eager readers, the code in the notebook can be found in the following condensed snippet. However, we do suggest reading the notebook as it contains more insight into each element of the program.
```python
import pandas as pd
import palimpzest.datamanager.datamanager as pzdm
from palimpzest.sets import Dataset
from palimpzest.core.lib.fields import Field
from palimpzest.core.lib.schemas import Schema, TextFile
from palimpzest.policy import MinCost, MaxQuality
from palimpzest.query.processor.config import QueryProcessorConfig

# Dataset registration
dataset_path = "testdata/enron-tiny"
dataset_name = "enron-tiny"
pzdm.DataDirectory().register_local_directory(dataset_path, dataset_name)

# Dataset loading
dataset = Dataset(dataset_name, schema=TextFile)

# Schema definition for the fields we wish to compute
class Email(Schema):
    """Represents an email, which in practice is usually from a text file"""
    sender = Field(desc="The email address of the sender")
    subject = Field(desc="The subject of the email")
    date = Field(desc="The date the email was sent")

# Lazy construction of computation to filter for emails about holidays sent in July
dataset = dataset.convert(Email, desc="An email from the Enron dataset")
dataset = dataset.filter("The email was sent in July")
dataset = dataset.filter("The email is about holidays")

# Executing the compuation
policy = MinCost()
config = QueryProcessorConfig(
    policy=policy,
    verbose=True,
    processing_strategy="no_sentinel",
    execution_strategy="sequential",
    optimizer_strategy="pareto",
)
results, execution_stats = dataset.run(config)

# Writing output to disk
output_df = pd.DataFrame([r.to_dict() for r in results])[["date","sender","subject"]]
output_df.to_csv("july_holiday_emails.csv")
```

## Palimpzest CLI
Installing Palimpzest also installs its CLI tool `pz` which provides users with basic utilities at the command line for creating and managing their own Palimpzest system. Please read the readme in [src/cli/README.md](./src/cli/README.md) for instructions on how to use it.

## Python Demos
Below are simple instructions to run PZ on a test data set of enron emails that is included with the system.

### Downloading test data
To run the provided demos, you will need to download the test data. Due to the size of the data, we are unable to include it in the repository. You can download the test data by running the following command from a unix terminal (requires `wget` and `tar`):
```
chmod +x testdata/download-testdata.sh
./testdata/download-testdata.sh
```
For convenience, we have also provided a script to register all test data with Palimpzest:
```
chmod +x testdata/register-sources.sh
./testdata/register-sources.sh
```

### Running the Demos
- Initialize the configuration by running `pz init`.

- Palimpzest defaults to using OpenAI. You’ll need to export an environment variable `OPENAI_API_KEY`

- (Skip this step if you ran the `register-sources.sh` script successfully) Add the enron data set with:
`pz reg --path testdata/enron-tiny --name enron-tiny`

- Finally, run the simple test program with:
      `python demos/simpleDemo.py --task enron --datasetid enron-eval-tiny --verbose`
