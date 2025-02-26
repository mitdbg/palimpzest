import argparse
import json
import os

import palimpzest as pz
import pandas as pd

from pneuma import Pneuma

question_dataset_cols = [
    {
        "name": "question",
        "type": str,
        "desc": "The question related to the contents of some table(s).",
    }
]

output_cols = [
    {
        "name": "answer",
        "type": str,
        "desc": "Output the answer to the `question` based on the contents in `relevant_tables`; output unknown if the contents cannot answer the question.",
    }
]


class QuestionDataReader(pz.DataReader):
    def __init__(self, questions_dataset_path: str, num_questions_to_process: int):
        super().__init__(question_dataset_cols)

        # `questions_dataset_path` is the path to the file containing the questions which is expected to be a jsonl file.
        # Each line in the file is a JSON object with an "id" and a "question" field.

        self.questions_dataset_path = questions_dataset_path
        self.num_questions_to_process = num_questions_to_process

        with open(questions_dataset_path) as f:
            entries = [json.loads(line) for line in f]
            entries = entries[: min(num_questions_to_process, len(entries))]
            self.questions = [entry["question"] for entry in entries]
            self.ids = [entry["id"] for entry in entries]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx: int):
        # get question
        question = self.questions[idx]

        # construct and return dictionary with field(s)
        return {"question": question}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Pneuma demo")
    parser.add_argument(
        "--questions-dataset-path",
        type=str,
        help="Path to the questions dataset",
        default="data_src/questions.jsonl",
    )
    parser.add_argument(
        "--num-questions-to-process",
        type=int,
        help="Number of questions from the dataset to process",
        default=5,
    )
    parser.add_argument(
        "--out-path", type=str, help="Path to the output file of Pneuma", default="pneuma-demo"
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Number of relevant documents to retrieve from the index; Pneuma returns <= k tables (each table may be associated with multiple documents)",
        default=5,
    )
    return parser.parse_args()


def build_pneuma_query(pneuma: Pneuma, dataset: QuestionDataReader, k: int):
    questions = pz.Dataset(dataset)
    questions = questions.sem_add_columns(
        question_dataset_cols, desc="Extract the question"
    )

    def extract_table_content(df: pd.DataFrame):
        columns = " | ".join(df.columns)
        rows = "\n".join(" | ".join(map(str, row[1:])) for row in df.itertuples())
        return f"{columns}\n{rows}"

    def search_func(index: Pneuma, query: str, k: int):
        response = index.query_index(
            index_name="demo_index",
            queries=query,
            k=k,
            n=5,
            alpha=0.5,
        )
        response = json.loads(response)
        retrieved_tables = response["data"][0]["retrieved_tables"]

        relevant_tables: list[str] = []
        for table in retrieved_tables:
            table_content = extract_table_content(pd.read_csv(table))
            relevant_tables.append(table_content)
        return relevant_tables

    questions_and_relevant_files = questions.retrieve(
        index=pneuma,
        search_func=search_func,
        search_attr="question",
        output_attr="relevant_tables",
        output_attr_desc="Most relevant tables to the `question`",
        k=k,
    )
    output = questions_and_relevant_files.sem_add_columns(output_cols)
    return output


def main():
    if os.getenv("OPENAI_API_KEY") is None and os.getenv("TOGETHER_API_KEY") is None:
        print("WARNING: Both OPENAI_API_KEY and TOGETHER_API_KEY are unset")

    args = parse_arguments()

    # Create a data reader for the question dataset (in this demo ChEMBL)
    dataset = QuestionDataReader(
        questions_dataset_path=args.questions_dataset_path,
        num_questions_to_process=args.num_questions_to_process,
    )

    # Load the index
    pneuma = Pneuma(
        out_path=args.out_path,
        llm_path="Qwen/Qwen2.5-7B-Instruct",  # Change to a smaller LLM if necessary
        embed_path="BAAI/bge-base-en-v1.5",
    )
    pneuma.setup()

    # Use OpenAI models (gpt-4o-mini & text-embedding-3-small) if the index was
    # generated using OpenAI models
    # pneuma = Pneuma(
    #     out_path=args.out_path,
    #     openai_api_key=os.environ['OPENAI_API_KEY'],
    #     use_local_model=False,
    # )

    # Build and run some query
    query = build_pneuma_query(pneuma, dataset, args.k)
    data_record_collection = query.run(pz.QueryProcessorConfig())
    print(data_record_collection.to_df())


if __name__ == "__main__":
    main()
