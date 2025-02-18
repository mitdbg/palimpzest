![pz-banner](https://palimpzest-workloads.s3.us-east-1.amazonaws.com/palimpzest-cropped.png)

# Palimpzest (PZ)
[![Discord](https://img.shields.io/discord/1245561987480420445?logo=discord)](https://discord.gg/dN85JJ6jaH)
[![Docs](https://img.shields.io/badge/Read_the_Docs-purple?logo=readthedocs)](https://palimpzest.org/)
[![Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zqOxnh_G6eZ8_xax6PvDr-EjMt7hp4R5?usp=sharing)
[![PyPI](https://img.shields.io/pypi/v/palimpzest)](https://pypi.org/project/palimpzest/)
[![PyPI - Monthly Downloads](https://img.shields.io/pypi/dm/palimpzest?color=teal)](https://pypi.org/project/palimpzest/)
<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv)](https://arxiv.org/pdf/2405.14696) -->
<!-- [![Video](https://img.shields.io/badge/YouTube-Talk-red?logo=youtube)](https://youtu.be/T8VQfyBiki0?si=eiph57DSEkDNbEIu) -->

## Learn How to Use PZ
Our [full documentation](https://palimpzest.org) is the definitive resource for learning how to use PZ. It contains all of the installation and quickstart materials on this page, as well as user guides, full API documentation, and much more.

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

## Join the PZ Community
We are actively hacking on PZ and would love to have you join our community [![Discord](https://img.shields.io/discord/1245561987480420445?logo=discord)](https://discord.gg/dN85JJ6jaH)

[Our Discord server](https://discord.gg/dN85JJ6jaH) is the best place to:
- Get help with your PZ program(s)
- Give feedback to the maintainers
- Discuss the future direction(s) of the project
- Discuss anything related to data processing with LLMs!

We are eager to learn more about your workloads and use cases, and will take them into consideration in planning our future roadmap.

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
import palimpzest as pz

# define the fields we wish to compute
email_cols = [
    {"name": "sender", "type": str, "desc": "The email address of the sender"},
    {"name": "subject", "type": str, "desc": "The subject of the email"},
    {"name": "date", "type": str, "desc": "The date the email was sent"},
]

# lazily construct the computation to get emails about holidays sent in July
dataset = pz.Dataset("testdata/enron-tiny/")
dataset = dataset.sem_add_columns(email_cols)
dataset = dataset.sem_filter("The email was sent in July")
dataset = dataset.sem_filter("The email is about holidays")

# execute the computation w/the MinCost policy
config = pz.QueryProcessorConfig(policy=pz.MinCost(), verbose=True)
output = dataset.run(config)

# display output (if using Jupyter, otherwise use print(output_df))
output_df = output.to_df(cols=["date", "sender", "subject"])
display(output_df)
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
