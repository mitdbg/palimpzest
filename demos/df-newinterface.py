import pandas as pd

import palimpzest as pz

df = pd.read_csv("testdata/enron-tiny.csv")
ds = pz.Dataset(df)
ds = ds.sem_add_columns([
    {"name" : "sender", "desc" : "The email address of the sender", "type" : str}, 
    {"name" : "subject", "desc" : "The subject of the email", "type" : str},
    {"name" : "date", "desc" : "The date the email was sent", "type" : str}
])

ds = ds.sem_filter("It is an email").sem_filter("It has Vacation in the subject")
output = ds.run()
output_df = output.to_df()
print(output_df)

output_df = output.to_df(project_cols=["sender", "subject", "date"])
print(output_df)
