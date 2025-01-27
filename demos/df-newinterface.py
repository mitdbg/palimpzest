import pandas as pd
import palimpzest as pz
from palimpzest.query.processor.config import QueryProcessorConfig
from palimpzest.core.elements.records import DataRecord

df = pd.read_csv("testdata/enron-tiny.csv")
qr2 = pz.Dataset(df)
qr2 = qr2.add_columns({"sender": ("The email address of the sender", "string"), 
                        "subject": ("The subject of the email", "string"),#
                        "date": ("The date the email was sent", "string")})
qr3 = qr2.filter("It is an email").filter("It has Vacation in the subject")

# config = QueryProcessorConfig(
#     verbose=False,

#     execution_strategy="pipelined_parallel",
# )

records, _ = qr3.run()
outputDf = DataRecord.to_df(records)



print(outputDf)
