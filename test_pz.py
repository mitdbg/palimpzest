import palimpzest as pz
from dotenv import load_dotenv

load_dotenv()
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
output_df = output.to_df(cols=["date", "sender", "subject"])
print(output_df)
