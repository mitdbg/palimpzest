import os
import pandas as pd
import pdb
from sklearn.metrics import precision_recall_fscore_support

IN_DIR= "testdata/biofabric-matching/"
RESULT_PATH = "results/biofabric/"

# def evaluate_matching():

output = pd.read_csv(os.path.join(RESULT_PATH, "clean_output.csv"))
index = [x for x in output.columns if x != "study"]

target_matching = pd.read_csv(os.path.join(RESULT_PATH, "target_matching.csv"), index_col=0).reindex(index)


studies = output["study"].unique()
# Group by output by the "study" column and split it into many dataframes indexed by the "study" column
df = pd.DataFrame(columns=target_matching.columns, index = index)
cols = output.columns
for study in studies:
    output_study = output[output["study"] == study]
    input_df = pd.read_excel(os.path.join(IN_DIR, f"{study}.xlsx"))

    # for every column in output_study, check which column in input_df is the closest, i.e. the one with the highest number of matching values
    for col in cols:
        if col == "study":
            continue
        max_matches = 0
        max_col = "missing"
        for input_col in input_df.columns:
            try:
                matches = sum([1 for idx,x in enumerate(output_study[col]) if x == input_df[input_col]
                [idx]])
            except:
                pdb.set_trace()
            if matches > max_matches:
                max_matches = matches
                max_col = input_col
        df.loc[col, study] = max_col

        # build a matrix that has the study on the columns and the predicted column names on the rows
    df.fillna("missing", inplace=True)   
    p,r,f1,sup = precision_recall_fscore_support(target_matching[study].values, df[study].values, average="weighted", zero_division=0)
    print(f"Study {study} has an F1 score of {f1}, Precision of {p} and Recall of {r}")


print(df)


# if __name__ == "__main__":
#     evaluate_matching()