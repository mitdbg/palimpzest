![pz-banner](logos/palimpzest-cropped.png)

# Palimpzest (PZ)
- **Read our (pre-print) paper:** [**read the paper**](https://arxiv.org/pdf/2405.14696)
- Join our Discord: [discord](https://discord.gg/znFN2baN)
- Read our short blog post: [read the blog post](https://dsg.csail.mit.edu/projects/palimpzest/)
- Check out our Colab Demo: [colab demo](https://colab.research.google.com/drive/1zqOxnh_G6eZ8_xax6PvDr-EjMt7hp4R5?usp=sharing)
- Check out the video: [MIT 2024](https://youtu.be/T8VQfyBiki0?si=eiph57DSEkDNbEIu)

# Getting started
You can install the Palimpzest package and CLI on your machine by cloning this repository and running:
```bash
$ git clone git@github.com:mitdbg/palimpzest.git
$ cd palimpzest
$ pip install .
```

## Downloading test data
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


## Python Demos
#### NOTE: we are in the process of refactoring our demos; please reach out to us or create an issue if you'd like support getting started
Below are simple instructions to run pz on a test data set of enron emails that is included with the system:

- Initialize the configuration by running `pz init`.

- Palimpzest defaults to using OpenAI. You’ll need to export an environment variable `OPENAI_API_KEY`

- (Skip this step if you ran the `register-sources.sh` script successfully) Add the enron data set with:
`pz reg --path testdata/enron-tiny --name enron-tiny`

- Finally, run the simple test program with:
      `python demos/simpleDemo.py --task enron --datasetid enron-eval-tiny --verbose`

- If you would like to try running our in our execution mode which first optimizes on a subset of the data, you can run:
      `python demos/optimizerDemo.py --verbose --workload enron --datasetid enron-eval-tiny --executor sequential-mab --sample-budget 15 --exp-name demo` 



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

Users can initialize their own system by running `pz init`. This will create Palimpzest's working directory in `~/.palimpzest`:
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

### Registering Datasets
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

### Cache Management
Palimpzest will cache intermediate results by default. It can be useful to remove them from the cache when trying to evaluate the performance improvement(s) of code changes. We provide a utility command `pz clear-cache` (also aliased as `pz clr`) to clear the cache:
```bash
$ pz clr
Cache cleared
```

### Config Management
You may wish to work with multiple configurations of Palimpzest in order to, e.g., evaluate the difference in performance between various LLM services for your data extraction task. To see the config Palimpzest is currently using, you can run the `pz print-config` command (also aliased as `pz config`):
```bash
$ pz config
--- default ---
filecachedir: /some/local/filepath
llmservice: openai
name: default
parallel: false
```
By default, Palimpzest uses the configuration named `default`. As shown above, if you run a script using Palimpzest out-of-the-box, it will use OpenAI endpoints for all of its API calls.

Now, let's say you wanted to try using [together.ai's](https://www.together.ai/) for your API calls, you could do this by creating a new config with the `pz create-config` command (also aliased as `pz cc`):
```bash
$ pz cc --name together-conf --llmservice together --parallel True --set
Created and set config: together-conf
```
The `--name` parameter is required and specifies the unique name for your config. The `--llmservice` and `--parallel` options specify the service to use and whether or not to process files in parallel. Finally, if the `--set` flag is present, Palimpzest will update its current config to point to the newly created config.

We can confirm that Palimpzest checked out our new config by running `pz config`:
```bash
$ pz config
--- together-conf ---
filecachedir: /some/local/filepath
llmservice: together
name: together-conf
parallel: true
```

You can switch which config you are using at any time by using the `pz set-config` command (also aliased as `pz set`):
```bash
$ pz set --name default
Set config: default

$ pz config
--- default ---
filecachedir: /some/local/filepath
llmservice: openai
name: default
parallel: false

$ pz set --name together-conf
Set config: together-conf

$ pz config
--- together-conf ---
filecachedir: /some/local/filepath
llmservice: together
name: together-conf
parallel: true
```

You can update an existing config using the `pz update` command (also aliased as `pz uc`):
```bash
$ pz update --name default --settings parallel=true,pdfprocessor=pdfplumber
Updated config: default

$ pz config
--- default ---
filecachedir: /some/local/filepath
llmservice: anthropic
name: default
parallel: true
pdfprocessor: pdfplumber
```

The `--name` parameter specifies which config to update. `--settings` specifies all the parameter name and value pairs in the format `param_name=param_value`, separated by commas.


Finally, you can delete a config with the `pz rm-config` command (also aliased as `pz rmc`):
```bash
$ pz rmc --name together-conf
Deleted config: together-conf
```
Note that you cannot delete the `default` config, and if you delete the config that you currently have set, Palimpzest will set the current config to be `default`.

## Configuring for Parallel Execution

There are a few things you need to do in order to use remote parallel services.

If you want to use parallel LLM execution on together.ai, you have to modify the config.yaml (by default, Palimpzest uses `~/.palimpzest/config_default.yaml`) so that `llmservice: together` and `parallel: True` are set.

If you want to use parallel PDF processing at modal.com, you have to:
1. Set `pdfprocessor: modal` in the config.yaml file.
2. Run `modal deploy src/palimpzest/tools/allenpdf.py`.  This will remotely install the modal function so you can run it. (Actually, it's probably already installed there, but do this just in case.  Also do it if there's been a change to the server-side function inside that file.)

