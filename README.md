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
You can install the Palimpzest package and CLI on your machine by cloning this repository and running:
```bash
$ git clone git@github.com:mikecafarella/palimpzest.git
$ cd palimpzest
$ pip install .
```

## Setting PZ_DIR
Palimpzest uses the environment variable `PZ_DIR` to set the root of its working directory. If this environment variable is not set, Palimpzest will create its working directory at `~/.palimpzest` by default. The CLI also allows you to override `PZ_DIR` with the `--pz-dir` flag when initializing the system (e.g. `pz init --pz-dir path/to/dir`).



## Palimpzest CLI
Installing Palimpzest also installs its CLI tool `pz` which provides users with basic utilities for creating and managing their own Palimpzest system. Running `pz --help` diplays an overview of the CLI's commands:
```bash
$ pz --help
Usage: pz [OPTIONS] COMMAND [ARGS]...

  The CLI tool for Palimpzest.

Options:
  --help  Show this message and exit.

Commands:
  help (h)                        Print the help message for PZ.
  init (i)                        Initialize data directory for PZ.
  ls-data (ls,lsdata)             Print a table listing the datasets
                                  registered with PZ.
  register-data (r,reg,register)  Register a data file or data directory with
                                  PZ.
  rm-data (rm,rmdata)             Remove a dataset that was registered with
                                  PZ.
```

Users can initialize their own system by running `pz init`. This will create Palimpzest's working directory in `~/.palimpzest` (unless `PZ_DIR` is set, or `--pz-dir` is specified):
```bash
$ pz init
Palimpzest system initialized in: /Users/matthewrusso/.palimpzest
```

If we list the set of datasets registered with Palimpzest, we'll see there currently are none:
```bash
$ pz ls
+------+------+------+
| Name | Type | Path |
+------+------+------+
+------+------+------+

Total datasets: 0
```

To add (or "register") a dataset with Palimpzest, we can use the `pz register-data` command (also aliased as `pz reg`) to specify that a file or directory at a given `--path` should be registered as a dataset with the specified `--name`:
```bash
$ pz reg --path README.md --name rdme
Registered rdme
```

If we list Palimpzest's datasets again we will see that `README.md` has been registered under the dataset named `rdme`:
```bash
$ pz ls
+------+------+------------------------------------------+
| Name | Type |                   Path                   |
+------+------+------------------------------------------+
| rdme | file | /Users/matthewrusso/palimpzest/README.md |
+------+------+------------------------------------------+

Total datasets: 1
```

To remove a dataset from Palimpzest, simply use the `pz rm-data` command (also aliased as `pz rm`) and specify the `--name` of the dataset you would like to remove:
```bash
$ pz rm --name rdme
Deleted rdme
```

Finally, listing our datasets once more will show that the dataset has been deleted:
```bash
$ pz ls
+------+------+------+
| Name | Type | Path |
+------+------+------+
+------+------+------+

Total datasets: 0
```

## Configuring for parallel execution

There are a few things you need to do in order to use remote parallel services.

If you want to use parallel LLM execution on together.ai, you have to modify the config.yaml so that `llmservice: together` and `parallel: True` are set.

If you want to use parallel PDF processing at modal.com, you have to:
1. Set `pdfprocessing: modal` in the config.yaml file.
2. Run `modal deploy src/palimpzest/tools/allenpdf.py`.  This will remotely install the modal function so you can run it. (Actually, it's probably already installed there, but do this just in case.  Also do it if there's been a change to the server-side function inside that file.)



## Python Demo

Below are simple instructions to run pz on a test data set of enron emails that is included with the system:

- Set the system environment variables `PZ_DIR`. This is the root directory for the Palimpzest system.

- Add the pz tool to your path (it is in the tools directory).  

`export PATH=$PATH:$PZ_DIR/tools/`

- Initialize the configuration by running `pz --init`.

- Add the enron data set with:
`pz reg --path testdata/enron-tiny --name enron-tiny`
then run it through the test program with:
      `tests/simpleDemo.py --task enron --datasetid enron-tiny`

- Add the test paper set with:
    `pz reg --path testdata/pdfs-tiny --name pdfs-tiny`
then run it through the test program with:
`tests/simpleDemo.py --task paper --datasetid pdfs-tiny`


- Palimpzest defaults to using OpenAI. You’ll need to export an environment variable `OPENAI_API_KEY`


