# Simple and Powerful Semantic Computation
[![Discord](https://img.shields.io/discord/1245561987480420445?logo=discord)](https://discord.gg/dN85JJ6jaH)
[![Colab Demo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zqOxnh_G6eZ8_xax6PvDr-EjMt7hp4R5?usp=sharing)
[![PyPI](https://img.shields.io/pypi/v/palimpzest)](https://pypi.org/project/palimpzest/)
[![PyPI - Monthly Downloads](https://img.shields.io/pypi/dm/palimpzest?color=teal)](https://pypi.org/project/palimpzest/)
[![GitHub](https://img.shields.io/badge/GitHub-Code-blue?logo=github)](https://github.com/mitdbg/palimpzest)
<!-- [![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv)](https://arxiv.org/pdf/2405.14696) -->
<!-- [![Video](https://img.shields.io/badge/YouTube-Talk-red?logo=youtube)](https://youtu.be/T8VQfyBiki0?si=eiph57DSEkDNbEIu) -->

Palimpzest (PZ) enables developers to write simple, powerful programs which use semantic operators (i.e. LLMs) to perform computation.

The following code snippet sets up PZ and downloads a small datast of emails:
```bash
# setup in the terminal
$ pip install palimpzest
$ export OPENAI_API_KEY="<your-api-key>"
$ wget https://palimpzest-workloads.s3.us-east-1.amazonaws.com/emails.zip
$ unzip emails.zip
```
We can then execute a simple PZ program to:

1. compute the `subject` and `date` of each email
2. filter for emails about vacations which are sent in July

```python
import palimpzest as pz

emails = pz.Dataset("emails/")
emails = emails.sem_add_columns([
    {"name": "subject", "type": str, "desc": "the subject of the email"},
    {"name": "date", "type": str, "desc": "the date the email was sent"},
])
emails = emails.sem_filter("The email is about vacation")
emails = emails.sem_filter("The email was sent in July")
output = emails.run(max_quality=True)

print(output.to_df(cols=["filename", "date", "subject"]))
```
The output from this program is shown below:
```
     filename         date                  subject
0  email4.txt   6 Jul 2001           Vacation plans
1  email5.txt  26 Jul 2001  Vacation Days in August
```

### Key Features of PZ
There are a few features of this program which are worth highlighting:

1. The programmer creates a `pz.Dataset` from the directory of emails and defines a series of ***semantic computations*** on that dataset:
    - `sem_add_columns()` specifies a set of fields which PZ must compute
    - `sem_filter()` selects for emails which satisfy the natural language filter
2. The user does not specify ***how*** the computation should be performed -- they simply declare ***what*** they want PZ to compute
    - This is what makes PZ declarative
3. Under the hood, PZ's optimizer determines the best way to execute each **semantic operator**
    - In this example, PZ optimizes for output quality because the user sets `max_quality=True`
3. The `output` is not generated until the call to `emails.run()`
    - i.e. PZ uses [lazy evaluation](https://en.wikipedia.org/wiki/Lazy_evaluation)

### Declarative Optimization for AI
The core philosophy behind PZ is that programmers should simply specify the high-level logic of their AI programs while offloading much of the performance tuning to a powerful optimizer. Of course, users still have the ability to fully control their program, and can override and assist the optimizer (if needed) to get the best possible performance.

This email processing example only showcases a small set of the semantic operators implemented in PZ. Other operators include:

- `retrieve()` which takes a vector database and a search string as input and retrieves the most relevant entries from the database
- `add_columns()` and `filter()` which are the non-semantic equivalents of `sem_add_columns()` and `sem_filter()`
- `groupby()`, `count()`, `average()`, `limit()`, and `project()` which mirror their implementations in frameworks like Pandas and Spark.

<!-- **[[start of quick editorial note]]**
This^ example is significantly improved from before, but it can be simplified and clarified further:
0. Add `maxquality=True` as a flag for `.run()` (i.e. make it possible to construct config from `.run()` kwargs)
1. If possible, we should show the (abbreviated) contents of the inputs
2. As discussed offline, the `sem_add_columns()` arguments are very verbose, and it would be nice to support (and show off) syntax like:
    - `emails.sem_add_columns(["sender", "subject"], prompt="Please compute the subject and sent date of the email")`.
3. I also think we should also consider making our example a bit more impressive.
    - Right now it feels like a childish example which a few regexes could solve
    - We need to be showcasing PZ's muscle right from the jump (e.g. computing a summary of the email)
4. A good solution to 5. can also solve (1.) - (4.)
**[[end of quick editorial note]]** -->

<!-- PZ provides the developer with a high-level interface for composing semantic operators into concise programs. The call to `emails.run()` triggers PZ's optimizer, which automatically selects which LLMs and execution strategies to use for each semantic operation. Users have the ability to fully control the program, and can override and assist the optimizer (if needed) to get the best possible performance. -->

### Join our community
We strongly encourage you to join our [Discord server](https://discord.gg/dN85JJ6jaH) where we are happy to help you get started with PZ.

### What's Next?
The rest of our Getting Started section will:

1. Help you install PZ
2. Explore more of PZ's features in our [Quick Start Tutorial](getting-started/quickstart.md)
3. Give you an overview of our [User Guides](user-guide/overview.md) which discuss features of PZ in more depth


<!-- Palimpzest is a **cost-based optimizer for AI-powered analytical workloads**. It enables users to express complex AI-powered data queries in a **high-level declarative language**, and it **automatically generates optimized execution plans** that minimize cost, maximize quality, or balance both.

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
        from palimpzest.core.data.dataset import Dataset
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
            `python demos/simpleDemo.py --task enron --datasetid enron-eval-tiny --verbose` -->
