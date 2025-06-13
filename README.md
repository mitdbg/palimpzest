![pz-banner](https://palimpzest-workloads.s3.us-east-1.amazonaws.com/palimpzest-cropped.png)

# Palimpzest (PZ)
[![Discord](https://img.shields.io/discord/1245561987480420445?logo=discord)](https://discord.gg/dN85JJ6jaH)
[![Docs](https://img.shields.io/badge/Read_the_Docs-purple?logo=readthedocs)](https://palimpzest.org/)
[![Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Fm8I4yL1az395MsFkQbEIZSmUZs0oGvZ?usp=sharing)
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

## Python Demos
Below are simple instructions to run PZ on a test data set of enron emails that is included with the system.

### Downloading test data
To run the provided demos, you will need to download the test data. Due to the size of the data, we are unable to include it in the repository. You can download the test data by running the following command from a unix terminal (requires `wget` and `tar`):
```
chmod +x testdata/download-testdata.sh
./testdata/download-testdata.sh
```

### Running the Demos
Set your OpenAI (or Together.ai) api key at the command line:
```bash
# set one (or both) of the following:
export OPENAI_API_KEY=<your-api-key>
export TOGETHER_API_KEY=<your-api-key>
```

Now you can run the simple test program with:
```bash
$ python demos/simple-demo.py --task enron --dataset testdata/enron-eval-tiny --verbose
```

### Citation
If you would like to cite our work, please use the following citation:
```
@inproceedings{palimpzestCIDR,
    title={Palimpzest: Optimizing AI-Powered Analytics with Declarative Query Processing},
    author={Liu, Chunwei and Russo, Matthew and Cafarella, Michael and Cao, Lei and Chen, Peter Baile and Chen, Zui and Franklin, Michael and Kraska, Tim and Madden, Samuel and Shahout, Rana and Vitagliano, Gerardo},
    booktitle = {Proceedings of the {{Conference}} on {{Innovative Database Research}} ({{CIDR}})},
    date = 2025,
}
```