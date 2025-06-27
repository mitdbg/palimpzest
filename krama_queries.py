import os

import palimpzest as pz


def legal_easy_3():
    # legal-easy-*
    ds = pz.TextFileContext(
        path="../Kramabench/data/legal/",
        id="legal-dataset",
        description="Files containing statistics from the FTC on identity theft, fraud, and other reports.",
    )
    # ds = ds.search(
    #     "information on the number of identity theft reports in 2024 and 2001",
    #     context="identity_theft_reports_2024_2021",
    # )
    ds = ds.compute(
        "the total number of identity theft reports in 2024 and in 2001",
        # use_context="identity_theft_reports_2024_2021",
    )
    ds = ds.compute(
        "the ratio of identity theft reports in 2024 vs 2001, rounded to 4 decimal places",
        # use_context="identity_theft_reports_2024_2021",
        # type=float,
    )
    out = ds.run(processing_strategy="no_sentinel", progress=False)
    out.to_df().to_csv("krama-results/legal-easy-3.csv", index=False)

    return out.to_df().iloc[0][""] # TODO


if __name__ == "__main__":
    os.makedirs("krama-results", exist_ok=True)

    # first time
    legal_easy_3_answer = legal_easy_3()
    print(legal_easy_3_answer)

    # second time
    legal_easy_3_answer = legal_easy_3()
    print(legal_easy_3_answer)
