# Palimpzest
Palimpzest is a framework for writing document-centric programs. It will help you marshal, clean, extract, transform, and integrate documents and data. The LLM compute platform is going to read and write a lot of documents; Palimpzest is how programmers can control it.

Some nice things Palimpzest does for you:
- Write ETL-style programs very quickly. The code is reusable, composable, and shareable.
- Declarative data quality management: you focus on the app, and let the system figure out quality improvements. Don't bother with model details; let the compiler handle it.
- Declarative runtime platform management: you tell the system how much money you want to spend, and let the system figure out how to make the program as fast as possible.
- Automatic data marshaling. Data naming, sampling, and caching are first-class concepts
- Ancillary material comes for free: annotation tools, accuracy reports, data version updates, useful provenance records, etc

Some target use cases for Palimpzest:
- Extract a useable pandemic model from a scientific paper that is accompanied by its code and test datasets
- Extract all the data tuples from every experiment in every battery electrolyte paper ever written, then write a simple query on them 
- Integrate multimodal bioinformatics data and make a nice exploration tool
- Process all the footnotes in all the bank regulatory statements to find out which ones are in trouble
- Comb through historical maps to find likely deposits of critical minerals
- Create a dashboard to understand your software team's work. Integrate GitHub commits with bug reports and the next release's feature list
- Integrate your datacenter's logs with background documentation, then ask for hypotheses about a bug you're seeing in Datadog

# Getting started

First thing, run `tests/simpleDemo.py`

