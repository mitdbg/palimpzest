# Note: include in the tests folder a .env file that contains the API keys for the services used in the tests
import os

import palimpzest as pz

if not os.environ.get('OPENAI_API_KEY'):
    from palimpzest.utils import load_env
    load_env()

pz.DataDirectory().clearCache(keep_registry=True)

class CaseData(pz.Schema):
    """An individual row extracted from a table containing medical study data."""
    case_submitter_id = pz.Field(desc="The ID of the case", required=True)
    age_at_diagnosis = pz.Field(desc="The age of the patient at the time of diagnosis", required=False)
    race = pz.Field(desc="An arbitrary classification of a taxonomic group that is a division of a species.", required=False)
    ethnicity = pz.Field(desc="Whether an individual describes themselves as Hispanic or Latino or not.", required=False)
    gender = pz.Field(desc="Text designations that identify gender.", required=False)
    vital_status = pz.Field(desc="The vital status of the patient", required=False)
    primary_diagnosis = pz.Field(desc="Text term used to describe the patient's histologic diagnosis, as described by the World Health Organization's (WHO) International Classification of Diseases for Oncology (ICD-O).", required=False)

# Make sure to run
# pz reg --name biofabric-tiny --path testdata/biofabric-tiny
pz.DataDirectory().clearCache(keep_registry=True)

xls = pz.Dataset('biofabric-tiny', schema=pz.CSVFile)
patient_tables = xls.convert(pz.Table, desc="All tables in the file", cardinality="oneToMany")
patient_tables = patient_tables.filter("The table contains biometric information about the patient")
case_data = patient_tables.convert(CaseData, desc="The patient data in the table",cardinality="oneToMany")

output = patient_tables

policy = pz.MinCost()
engine = pz.StreamingSequentialExecution

tables, plan, stats = pz.Execute(patient_tables, 
                                  policy = policy,
                                  nocache=True,
                                  execution_engine=engine)

for table in tables:
    header = table.header
    subset_rows = table.rows[:3]

    print("Table name:", table.name)
    print(" | ".join(header)[:100], "...")
    for row in subset_rows:
        print(" | ".join(row)[:100], "...")
    print()

print(stats)