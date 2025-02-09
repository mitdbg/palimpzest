# Getting Started

![Palimpzest](logos/palimpzest-cropped.png){ width="200", align=left }

# _Optimize_ _AI_ _Workloads_ _Declaratively_


Palimpzest is a **cost-based optimizer for AI-powered analytical workloads**. It enables users to express complex AI-powered data queries in a **high-level declarative language**, and it **automatically generates optimized execution plans** that minimize cost, maximize quality, or balance both.

In modern AI applications, executing queries efficiently is a challenge. A single query may require:

* Extracting structured data from unstructured sources (e.g., PDFs, emails, research papers)
* **Choosing between different AI models and inference methods**
* **Managing trade-offs between execution speed, cost, and accuracy**
* **Handling large-scale datasets while minimizing computational overhead**

Traditionally, AI engineers must **manually fine tune** prompts, select models, and optimize inference strategies for each task. This process is not only time consuming but also requires constant updates as models evolve and costs fluctuate.

Palimpzest **solves this problem** by applying **cost-based optimization techniques** similar to a database query optimizer to **AI-powered analytical queries**. Users write **declarative queries**, and Palimpzest:

1. **Analyzes the query structure**  
2. **Explores different execution plans**  
3. **Estimates cost, runtime, and quality**  
4. **Selects the optimal plan** based on user-defined constraints  

🚀 **Quick Links**:

- **[📄 Read the Paper](https://arxiv.org/pdf/2405.14696)**
- **[📝 Read the Blog](https://dsg.csail.mit.edu/projects/palimpzest/)**
- **[💬 Join the Discord](https://discord.gg/znFN2baN)**
- **[▶️ Watch the MIT Video](https://youtu.be/T8VQfyBiki0?si=eiph57DSEkDNbEIu)** 


!!! info "Getting Started I: Install Palimpzest"
    === "PyPi"
        You can find a stable version of the PZ package on PyPI [here](https://pypi.org/project/palimpzest/). To install the package, run:
        ```bash
        $ pip install palimpzest
        ```
    === "Clone Repo"
        Clone the repository and install the package:

        ```bash 
        git clone git@github.com:mitdbg/palimpzest.git
        cd palimpzest
        pip install .
        ```

!!! info "Getting Started II: Demo PZ modules for various tasks"

    === "Quick Start"

        The easiest way to get started with Palimpzest is to run the `quickstart.ipynb` jupyter notebook. We demonstrate the full workflow of working with PZ, including registering a dataset, composing and executing a pipeline, and accessing the results.
        To run the notebook, you can use the following command:
            ```bash
            $ jupyter notebook
            ```
        And then access the notebook from the jupyter interface in your browser at `localhost:8888`.

    === "Even Quicker Start"

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

    === "Python Demos"

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
