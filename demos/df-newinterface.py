import pandas as pd

import palimpzest as pz

df = pd.read_csv("testdata/enron-tiny.csv")
qr2 = pz.Dataset(df)
qr2 = qr2.sem_add_columns([
    {"name" : "sender", "desc" : "The email address of the sender", "type" : "string"}, 
    {"name" : "subject", "desc" : "The subject of the email", "type" : "string"},
    {"name" : "date", "desc" : "The date the email was sent", "type" : "string"}
])

qr3 = qr2.sem_filter("It is an email").sem_filter("It has Vacation in the subject")
output = qr3.run()
output_df = output.to_df()
print(output_df)

output_df = output.to_df(project_cols=["sender", "subject", "date"])
print(output_df)
