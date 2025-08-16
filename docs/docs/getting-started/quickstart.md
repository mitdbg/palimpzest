<!-- ## Goal
This page should expand upon the "teaser" code shown at the very beginning of `docs/index.md`.

Every user who reads this page should be left with the following impression:

1. PZ is easy to get started with
2. PZ is powerful: many (serious) AI programs can be written in PZ
3. PZ has an optimizer, which can help optimize the user's program
4. The user can interact with and override the optimizer's decisions

The ideal way to do this is with the old mantra "show don't tell." A great example application can start very small and simple (1.), then grow to be more complex (2.), then make a call to our optimizer (3.), after which a user inspects the plan and modifies certain aspects of it (4.).

(In this page, I don't expect us to do any more than call `plan = ds.optimize()` for demonstrating (3.). For now, we should only allude to more powerful optimization which we will talk about later in the docs). -->


### Creating a Dataset
Let's revisit our example from the [Getting Started](../index.md) page in more depth, starting with the first two lines:
```python
import palimpzest as pz

emails = pz.Dataset("emails/")
```
In this example, we provide `pz.Dataset`'s constructor with the path to a local directory as input. The directory has a flat structure, with one email per file:
```bash
emails
├── email1.txt
├── email2.txt
...
└── email9.txt
```
Given this flat directory, PZ will create a [`pz.IterDataset`](../user-guide/dataset.md), which iterates over the files in the directory at runtime.
??? note "What if my data isn't this simple?"

    That's perfectly fine!
    
    The `pz.IterDataset` class can be subclassed by the user to read data from more complex sources. The user just has to:
    
    1. implement the IterDataset's `__len__()` method
    2. implement the IterDataset's `__getitem__()` method
    
    More details can be found in our [user guide for custom Datasets](../user-guide/dataset.md).

The `pz.IterDataset` will emit one dictionary per file to the next operator in the program. By default, each dictionary will have two keys: `"contents"` and `"filename"` which map to the file's contents and filename, respectively:

```python
import palimpzest as pz

emails = pz.Dataset("emails/")
output = emails.run()

print(output.to_df())

# This produces the following output:
#                                             contents    filename
# 0  Message-ID: <1390685.1075853083264.JavaMail.ev...  email1.txt
# 1  Message-ID: <19361547.1075853083287.JavaMail.e...  email2.txt
#                                                  ...         ...
# 8  Message-ID: <22163131.1075859380492.JavaMail.e...  email9.txt
```
??? note "What is `output`?"

    The `output` in the program above has type [`pz.DataRecordCollection`](../api/data/datarecordcollection.md).
    
    This object contains:

    1. The data emitted by the PZ program
    2. The execution stats (i.e. cost, runtime, and quality metrics) for the entire program

    We expose the `pz.DataRecordCollection.to_df()` method to make it easy for users to get the output(s) of their program in a Pandas DataFrame. We will also expose other utility methods for processing execution statistics in the near future.

### Computing New Fields
A key feature of PZ is that it provides users with the ability to compute new fields using semantic operators. To compute new fields, users need to invoke the `sem_add_columns()` method with a list of dictionaries defining the field(s) the system should compute:
```python
emails = emails.sem_add_columns([
    {"name": "subject", "type": str, "desc": "the subject of the email"},
    {"name": "date", "type": str, "desc": "the date the email was sent"},
])
```
In order to fully define a field, each dictionary must have the following three keys:

1. `name`: the name of the field
2. `type`: the type of the field (one of `str`, `int`, `float`, `bool`, `list[str]`, ..., `list[bool]`)
3. `desc`: a short natural langague description defining what the field represents

PZ will then use one (or more) LLM(s) to generate the field for every input to the operator (i.e. each email in this example).

??? info "But what is `sem_add_columns()` actually doing to generate the field(s)?"

    It depends! (and this is where PZ's optimizer comes in handy)

    Depending on the difficulty of the task and your preferred optimization objective (e.g. `max_quality`) PZ will select one implementation from a set of `PhysicalOperators` to generate your field(s).

    PZ can choose from 1,000+ possible implementations of its `PhysicalOperators`. Each operator uses one (or more) LLMs and may use techniques such as RAG, Mixture-of-Agents, Critique and Refine, etc. to produce a final output.

    For a full list of `PhysicalOperators` in PZ, please consult our documentation on [Operators](../api/operators/physical.md).

### Filtering Inputs
PZ also provides users with the ability to filter inputs using natural language. In order to apply a semantic filter, users need to invoke the `sem_filter()` method with a natural language description of the critieria they are *selecting for*:
```python
emails = emails.sem_filter("The email is about vacation")
emails = emails.sem_filter("The email was sent in July")
```
These filters will keep all emails which discuss vaction(s) and which were sent in the month of July.

### Optimization and Execution
Finally, once we've defined our program in PZ, we can optimize and execute it in order to generate our output:
```python
output = emails.run(max_quality=True)
```
The `pz.Dataset.run()` method triggers PZ's execution of the program that has been defined by applying semantic operators to `emails`. The `run()` method also takes a number of keyword arguments which can configure the execution of the program.

In particular, users can specify one ***optimization objective*** and (optionally) one ***constraint***:

**Optimization objectives:**

- `max_quality=True` (maximize output quality) 
- `min_cost=True` (minimize program cost)
- `min_time=True` (minimize program runtime)

**Constraints:**

- `quality_threshold=<float>` (threshold in range [0, 1])
- `cost_budget=<float>` (cost in US Dollars)
- `time_budget=<float>` (time in seconds)

??? note "More Info on Constraints"

    PZ can only *estimate* the cost, quality, and runtime of each physical operator, therefore constraints are not guaranteed to be met. Furthermore, some constraints may be infeasible (even with perfect estimates).

    In any case, PZ will make a best effort attempt to find the optimal plan for your stated objective and constraint (if present).

    To achieve better estimates -- and thus better optimization outcomes -- please read our [Optimization User Guide](../user-guide/optimization.md).

In this example we do not provide validation data to PZ. Therefore, output quality is measured relative to the performance of a "champion model", i.e. the model with the highest MMLU score that is available to the optimizer.

In our [Optimization User Guide](../user-guide/optimization.md) we show you how to:

1. provide validation data to improve the optimizer's performance
2. override the optimizer if you wish to specify, for example, the specific model to use for a given operation

!!! info "Optimization: Design Philosophy"

    The optimizer is meant to help the programmer quickly get to a final program (i.e. a **plan**).

    In the best case, the optimizer can automatically select a plan that meets the developer's needs.
    
    However, in cases where it falls short, we try to make it as easy as possible for developers to iterate on changes to their plan until it achieves satisfactory performance.

### Examining Program Output
Finally, once your program finishes executing you can convert its output to a Pandas DataFrame and examine the results:
```python
print(output.to_df(cols=["filename", "date", "subject"]))
```
The `cols` keyword argument allows you to select which columns should populate your DataFrame (if it is `None`, then all columns are selected).

As mentioned in a note above, the `output` is a `pz.DataRecordCollection` which also contains all of the execution statistics for your program. We can use this to examine the total cost and runtime of our program:
```python
print(f"Total time: {output.execution_stats.total_execution_time:.1f}")
print(f"Total cost: {output.execution_stats.total_execution_cost:.3f}")
```
Which will produce an output like:
```
Total time: 41.7
Total cost: 0.081
```

### What's Next?
Click below to proceed to the `Next Steps`.
