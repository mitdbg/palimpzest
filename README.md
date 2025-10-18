![pz-banner](https://palimpzest-workloads.s3.us-east-1.amazonaws.com/palimpzest-cropped.png)

# Palimpzest (PZ)
[![Discord](https://img.shields.io/discord/1245561987480420445?logo=discord)](https://discord.gg/dN85JJ6jaH)
[![Docs](https://img.shields.io/badge/Read_the_Docs-purple?logo=readthedocs)](https://palimpzest.org/)
[![Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Fm8I4yL1az395MsFkQbEIZSmUZs0oGvZ?usp=sharing)
[![PyPI](https://img.shields.io/pypi/v/palimpzest)](https://pypi.org/project/palimpzest/)
[![PyPI - Monthly Downloads](https://img.shields.io/pypi/dm/palimpzest?color=teal)](https://pypi.org/project/palimpzest/)
<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv)](https://arxiv.org/pdf/2405.14696) -->
<!-- [![Video](https://img.shields.io/badge/YouTube-Talk-red?logo=youtube)](https://youtu.be/T8VQfyBiki0?si=eiph57DSEkDNbEIu) -->

## 📚 Learn How to Use PZ
Our [full documentation](https://palimpzest.org) is the definitive resource for learning how to use PZ. It contains all of the installation and quickstart materials on this page, as well as user guides, full API documentation (coming soon), and much more.

## 🚀 Getting started
You can find a stable version of the PZ package on PyPI [here](https://pypi.org/project/palimpzest/). To install the package, run:
```bash
$ pip install palimpzest
```

You can also install PZ with [uv](https://docs.astral.sh/uv/) for a faster installation:
```bash
$ uv pip install palimpzest
```

Alternatively, to install the latest version of the package from this repository, you can clone this repository and run the following commands:
```bash
$ git clone git@github.com:mitdbg/palimpzest.git
$ cd palimpzest
$ pip install .
```

## 🙋🏽 Join the PZ Community
We are actively hacking on PZ and would love to have you join our community [![Discord](https://img.shields.io/discord/1245561987480420445?logo=discord)](https://discord.gg/dN85JJ6jaH)

[Our Discord server](https://discord.gg/dN85JJ6jaH) is the best place to:
- Get help with your PZ program(s)
- Give feedback to the maintainers
- Discuss the future direction(s) of the project
- Discuss anything related to data processing with LLMs!

We are eager to learn more about your workloads and use cases, and will take them into consideration in planning our future roadmap.

### 📓 Citation
If you would like to cite our original paper on Palimpzest, please use the following citation:
```
@inproceedings{palimpzestCIDR,
    title={Palimpzest: Optimizing AI-Powered Analytics with Declarative Query Processing},
    author={Liu, Chunwei and Russo, Matthew and Cafarella, Michael and Cao, Lei and Chen, Peter Baile and Chen, Zui and Franklin, Michael and Kraska, Tim and Madden, Samuel and Shahout, Rana and Vitagliano, Gerardo},
    booktitle = {Proceedings of the {{Conference}} on {{Innovative Database Research}} ({{CIDR}})},
    date = 2025,
}
```

If you would like to cite our paper on Palimpzest's optimizer Abacus, please use the following citation:
```
@misc{russo2025abacuscostbasedoptimizersemantic,
      title={Abacus: A Cost-Based Optimizer for Semantic Operator Systems}, 
      author={Matthew Russo and Sivaprasad Sudhir and Gerardo Vitagliano and Chunwei Liu and Tim Kraska and Samuel Madden and Michael Cafarella},
      year={2025},
      eprint={2505.14661},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2505.14661}, 
}
```
