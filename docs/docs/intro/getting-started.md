## Goal
This page should expand upon the "teaser" code shown at the very beginning of `docs/index.md`.

Every user who reads this page should be left with the following impression:

1. PZ is easy to get started with
2. PZ is powerful: many (serious) AI programs can be written in PZ
3. PZ has an optimizer, which can help optimize the user's program
4. The user can interact with and override the optimizer's decisions

The ideal way to do this is with the old mantra "show don't tell." A great example application can start very small and simple (1.), then grow to be more complex (2.), then make a call to our optimizer (3.), after which a user inspects the plan and modifies certain aspects of it (4.).

(In this page, I don't expect us to do any more than call `plan = ds.optimize()` for demonstrating (3.). For now, we should only allude to more powerful optimization which we will talk about later in the docs).

---

### (copying contents of README as placeholder)
We demonstrate the workflow of working with PZ, ~~including registering a dataset~~, writing a program, executing the program, and accessing the results.

```python
import palimpzest as pz

# define the fields we wish to compute
email_cols = [
    {"name": "sender", "type": str, "desc": "The email address of the sender"},
    {"name": "subject", "type": str, "desc": "The subject of the email"},
    {"name": "date", "type": str, "desc": "The date the email was sent"},
]

# lazily construct the computation to get emails about holidays sent in July
dataset = pz.Dataset("testdata/enron-tiny/")
dataset = dataset.sem_add_columns(email_cols)
dataset = dataset.sem_filter("The email was sent in July")
dataset = dataset.sem_filter("The email is about holidays")

# execute the computation w/the MinCost policy
config = pz.QueryProcessorConfig(policy=pz.MinCost(), verbose=True)
output = dataset.run(config)

# display output (if using Jupyter, otherwise use print(output_df))
output_df = output.to_df(project_cols=["date", "sender", "subject"])
display(output_df)
```