![pz-banner](src/static/palimpzest-cropped.png)

# Palimpzest (PZ)
- **Read our (pre-print) paper:** [**read the paper**](https://arxiv.org/pdf/2405.14696)
- Join our Discord: [discord](https://discord.gg/znFN2baN)
- Read our short blog post: [read the blog post](https://dsg.csail.mit.edu/projects/palimpzest/)
- Check out our Colab Demo: [colab demo](https://colab.research.google.com/drive/1zqOxnh_G6eZ8_xax6PvDr-EjMt7hp4R5?usp=sharing)
- Check out the video: [MIT 2024](https://youtu.be/T8VQfyBiki0?si=eiph57DSEkDNbEIu)

## Getting started
You can find a stable version of the Palimpzest package on PyPI [here](https://pypi.org/project/palimpzest/). To install the package, run:
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
The easiest way to get started with Palimpzest, is to run the `quickstart.ipynb` jupyter notebook. We provide a simple use case to showcase the workflow of working with Palimpzest, including registering a dataset, running a workload, and accessing the results.
To run the notebook, you can use the following command:
```bash
$ jupyter notebook
```
And then access the notebook from the jupyter interface in your browser at `localhost:8888`.

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
