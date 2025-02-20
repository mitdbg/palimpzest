---
hide:
  - navigation
  - toc
---

Research Papers
===============

Palimpzest has been the source of a number of research papers. Here is a timeline of the papers along with their citations.

Winter 2025
-----------
**PalimpChat: A Chat Interface for Palimpzest** \[[arXiv](https://arxiv.org/abs/2502.03368)\]

!!! abstract

    Thanks to the advances in generative architectures and large language models, data scientists can now code pipelines of machine-learning operations to process large collections of unstructured data. Recent progress has seen the rise of declarative AI frameworks (e.g., Palimpzest, Lotus, and DocETL) to build optimized and increasingly complex pipelines, but these systems often remain accessible only to expert programmers. In this demonstration, we present PalimpChat, a chat-based interface to Palimpzest that bridges this gap by letting users create and run sophisticated AI pipelines through natural language alone. By integrating Archytas, a ReAct-based reasoning agent, and Palimpzest's suite of relational and LLM-based operators, PalimpChat provides a practical illustration of how a chat interface can make declarative AI frameworks truly accessible to non-experts.

    Our demo system is publicly available online. At SIGMOD'25, participants can explore three real-world scenarios--scientific discovery, legal discovery, and real estate search--or apply PalimpChat to their own datasets. In this paper, we focus on how PalimpChat, supported by the Palimpzest optimizer, simplifies complex AI workflows such as extracting and analyzing biomedical data.

```
@misc{liu2025palimpchatdeclarativeinteractiveai,
    title={PalimpChat: Declarative and Interactive AI analytics}, 
    author={Chunwei Liu and Gerardo Vitagliano and Brandon Rose and Matt Prinz and David Andrew Samson and Michael Cafarella},
    year={2025},
    eprint={2502.03368},
    archivePrefix={arXiv},
    primaryClass={cs.AI},
    url={https://arxiv.org/abs/2502.03368}, 
}
```

**SciVar: Enabling Optimized Scientific Discovery in 16 Lines of Palimpzest Code** \[[arXiv](https://arxiv.org/abs/2411.14569)\]

!!! abstract

    The global output of academic publications exceeds 5 million articles per year, making it difficult for humans to keep up with even a tiny fraction of scientific output. We need methods to navigate and interpret the artifacts -- texts, graphs, charts, code, models, and datasets -- that make up the literature. This paper evaluates various methods for extracting mathematical model variables from epidemiological studies, such as "infection rate (α)", "recovery rate (γ)", and "mortality rate (μ)". Variable extraction appears to be a basic task, but plays a pivotal role in recovering models from scientific literature. Once extracted, we can use these variables for automatic mathematical modeling, simulation, and replication of published results.
    
    We introduce a benchmark dataset comprising manually-annotated variable descriptions and variable values extracted from scientific papers. Based on this dataset, we present several baseline methods for variable extraction based on Large Language Models (LLMs) and rule-based information extraction systems. Our analysis shows that LLM-based solutions perform the best. Despite the incremental benefits of combining rule-based extraction outputs with LLMs, the leap in performance attributed to the transfer-learning and instruction-tuning capabilities of LLMs themselves is far more significant. This investigation demonstrates the potential of LLMs to enhance automatic comprehension of scientific artifacts and for automatic model recovery and simulation.

```
@misc{liu2024variableextractionmodelrecovery,
    title={Variable Extraction for Model Recovery in Scientific Literature}, 
    author={Chunwei Liu and Enrique Noriega-Atala and Adarsh Pyarelal and Clayton T Morrison and Mike Cafarella},
    year={2024},
    eprint={2411.14569},
    archivePrefix={arXiv},
    primaryClass={cs.IR},
    url={https://arxiv.org/abs/2411.14569}, 
}
```

Spring 2024
-----------
**Palimpzest: Optimizing AI-Powered Analytics with Declarative Query Processing** \[[CIDR'25](https://www.vldb.org/cidrdb/papers/2025/p12-liu.pdf)\] \[[arXiv](https://arxiv.org/abs/2405.14696)\]

!!! abstract

    A long-standing goal of data management systems has been to build systems which can compute quantitative insights over large collections of unstructured data in a cost-effective manner. Until recently, it was difficult and expensive to extract facts from company documents, data from scientific papers, or metrics from image and video corpora. Today’s models can accomplish these tasks with high accuracy. However, a programmer who wants to answer a substantive AI-powered query must orchestrate large numbers of models, prompts, and data operations. In this paper, we present PALIMPZEST, a system that enables programmers to pose AI-powered analytical queries over arbitrary collections of unstructured data in a simple declarative language. The system uses a cost optimization framework—which explores the search space of AI models, prompting techniques, and related foundation model optimizations. PALIMPZEST implements the query while navigating the trade-offs between runtime, financial cost, and output data quality. We introduce a novel language for AI-powered analytics tasks, the optimization methods that PALIMPZEST uses, and the prototype system itself. We evaluate PALIMPZEST on a real-world workload. Our system produces plans that are up to 3.3 x faster and 2.9 x cheaper than a baseline method when using a singlethread setup, while also achieving superior F1-scores. PALIMPZEST applies its optimizations automatically, requiring no additional work from the user.

```
@inproceedings{palimpzestCIDR,
    title={Palimpzest: Optimizing AI-Powered Analytics with Declarative Query Processing},
    author={Liu, Chunwei and Russo, Matthew and Cafarella, Michael and Cao, Lei and Chen, Peter Baile and Chen, Zui and Franklin, Michael and Kraska, Tim and Madden, Samuel and Shahout, Rana and Vitagliano, Gerardo},
    booktitle = {Proceedings of the {{Conference}} on {{Innovative Database Research}} ({{CIDR}})},
    date = 2025,
}
```