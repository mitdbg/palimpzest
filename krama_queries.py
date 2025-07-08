import os

import chromadb

import palimpzest as pz
from palimpzest.constants import PZ_DIR


def clear_cache():
    # remove context files
    context_dir = os.path.join(PZ_DIR, "contexts")
    for filename in os.listdir(context_dir):
        os.remove(os.path.join(context_dir, filename))

    # clear collection
    chroma_dir = os.path.join(PZ_DIR, "chroma")
    chroma_client = chromadb.PersistentClient(chroma_dir)
    chroma_client.delete_collection("contexts")


def legal_easy_3(iter):
    # legal-easy-*
    ds = pz.TextFileContext(
        path="../Kramabench/data/legal/",
        id="legal-dataset",
        description="Files containing statistics from the FTC on identity theft, fraud, and other reports.",
    )
    ds = ds.search(
        "information on the number of identity theft reports in 2024 and 2001",
        # context="identity_theft_reports_2024_2021",
    )
    # ds = ds.compute(
    #     "the total number of identity theft reports in 2024 and in 2001",
    #     # use_context="identity_theft_reports_2024_2021",
    # )
    ds = ds.compute(
        "the ratio of identity theft reports in 2024 vs 2001, rounded to 4 decimal places",
        # use_context="identity_theft_reports_2024_2021",
        # type=float,
    )
    ds = ds.sem_add_columns(
        cols=[{"name": "final_answer", "type": float, "desc": "The ratio of identity theft reports in 2024 vs 2001; output None if the answer cannot be determined"}]
    )
    out = ds.run(processing_strategy="no_sentinel", progress=False)
    out.to_df().to_csv(f"krama-results/legal-easy-3-{iter}.csv", index=False)

    return out.to_df().iloc[0]['final_answer']


def execute_legal_iso():
    """Execute the legal dataset with each question being processed in isolation."""
    clear_cache()
    pass


def execute_legal_cache():
    """Execute the legal dataset with each question being processed in sequence."""
    clear_cache()
    pass


if __name__ == "__main__":
    # create results dir
    os.makedirs("krama-results", exist_ok=True)

    # tmp: remove
    clear_cache()

    # execute the query
    legal_easy_3_answer = legal_easy_3(0)
