# Palimpzest
Palimpzest is a framework for writing document-centric programs. It will help you marshal, clean, extract, transform, and integrate documents and data. The LLM compute platform is going to read and write a lot of documents; Palimpzest is how programmers can control it.

Some nice things Palimpzest does for you:
- Write ETL-style programs very quickly. The code is reusable, composable, and shareable.
- Declarative data quality management: you focus on the app, and let the system figure out quality improvements. Don't bother with model details; let the compiler handle it.
- Declarative runtime platform management: you tell the system how much money you want to spend, and let the system figure out how to make the program as fast as possible.
- Automatic data marshaling. Data naming, sampling, and caching are first-class concepts
- Ancillary material comes for free: annotation tools, accuracy reports, data version updates, useful provenance records, etc

Some target use cases for Palimpzest:
- **Information Extraction**: Extract a useable pandemic model from a scientific paper that is accompanied by its code and test datasets
- **Scientific Discovery**: Extract all the data tuples from every experiment in every battery electrolyte paper ever written, then write a simple query on them 
- **Data Integration**: Integrate multimodal bioinformatics data and make a nice exploration tool
- **Document Processing**: Process all the footnotes in all the bank regulatory statements to find out which ones are in trouble
- **Data Mining (get it???)**: Comb through historical maps to find likely deposits of critical minerals
- **Digital Twins**: Create a system to understand your software team's work. Integrate GitHub commits, bug reports, and the next release's feature list into a single integrated view. Then add alerting, summaries, rankers, explorers, etc.
- **Next-Gen Dashboards**: Integrate your datacenter's logs with background documentation, then ask for hypotheses about a bug you're seeing in Datadog. Go beyond the ocean of 2d plots.

# Getting started

Below are simple instructions to run pz on a test data set of enron emails that is included with the system:

- Set the system environment variables `PZ_DIR`. This is the root directory for the Palimpzest system.

- Add the pz tool to your path (it is in the tools directory).  

`export PATH=$PATH:$PZ_DIR/tools/`

- Initialize the configuration by running `pz --init`.  You can also run pz via a command like 
`python3 tools/pz --int`.

- Add a the enron data set with:

`pz registerdatadir testdata/enron-tiny enron-tiny`

- Run it through the test program with:
    `tests/simpleDemo.py --task enron --datasetid enron-tiny`

- Palimpzest defaults to using OpenAI. You’ll need to export an environment variable `OPENAI_API_KEY`

This simple 
