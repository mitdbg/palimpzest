import argparse
import json
import os
import string
import time

import numpy as np
from openai import OpenAI

from palimpzest.constants import MODEL_CARDS, Cardinality, Model
from palimpzest.utils.generation_helpers import get_json_from_answer


def f1(preds: list | None, targets: list):
    if preds is None or len(targets) == 0:
        return 0.0

    tp, fp, fn = 0, 0, 0
    try:
        # compute recall of retrieved ids and return
        preds = [str(pred).lower() for pred in preds]
        targets = [str(target).lower() for target in targets]

        remove_tokens = [c for c in string.punctuation if c != "/"]
        for token in remove_tokens:
            preds = [pred.replace(token, "") for pred in preds]
            targets = [target.replace(token, "") for target in targets]

        for pred in preds:
            if pred in targets:
                tp += 1
            else:
                fp += 1
        for target in targets:
            if target not in preds:
                fn += 1

        # compute overall f1 score and return
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return f1

    except Exception:
        os.makedirs("mmqa-recall-eval-errors", exist_ok=True)
        ts = time.time()
        with open(f"mmqa-recall-eval-errors/error-{ts}.txt", "w") as f:
            f.write(str(preds))
        return 0.0


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="Run a simple demo")
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed used to initialize RNG for MAB sampling algorithm",
    )
    args = parser.parse_args()

    # create directory for profiling data
    os.makedirs("opt-profiling-data", exist_ok=True)
    seed = args.seed
    print(f"Running with seed: {seed}")

    # start time for processing
    start_time = time.time()

    # read the appropriate dataset
    dataset = []
    with open("testdata/MMQA_dev.jsonl") as f:
        for line in f:
            dict_line = json.loads(line)
            if "image" in dict_line["metadata"]["modalities"] and len(dict_line["metadata"]["modalities"]) > 1:
                dataset.append(dict_line)

    # shuffle the questions for the given seed
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(dataset)

    # trim to number of samples
    dataset = dataset[:100]

    # construct the prompt
    prompt = """You are an intelligent AI assistant designed to answer questions. Please answer the following question to the best of your ability based on your prior knowledge.
    Return your answer as a JSON list of strings.
    Do not include any additional context or an explanation in your answer, simply list the entities asked for by the question:

QUESTION: {question}

ANSWER: 
"""

    # iterate over the dataset and generate answers
    model_name = "gpt-4o-mini-2024-07-18"
    preds, total_cost = [], 0.0
    for idx, entry in enumerate(dataset):
        print(f"Processing entry {idx}")
        formatted_prompt = prompt.format(question=entry["question"])
        client = OpenAI()
        payload = {
            "model": model_name,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": formatted_prompt}],
        }
        completion = client.chat.completions.create(**payload)

        # compute total cost
        usd_per_input_token = MODEL_CARDS[model_name]["usd_per_input_token"]
        usd_per_output_token = MODEL_CARDS[model_name]["usd_per_output_token"]
        input_tokens = completion.usage.prompt_tokens
        output_tokens = completion.usage.completion_tokens
        total_cost += input_tokens * usd_per_input_token + output_tokens * usd_per_output_token

        # extract answer
        completion_text = completion.choices[0].message.content
        try:
            answer = get_json_from_answer(completion_text, Model.GPT_4o_MINI, Cardinality.ONE_TO_MANY)
        except:
            answer = [completion_text]
        preds.append(answer)

    # get total time
    total_time = time.time() - start_time

    # score the output
    scores = []
    for pred, entry in zip(preds, dataset):
        answers = entry["answers"]
        answer = [ans["answer"] for ans in answers]
        f1_score = f1(pred, answer)
        scores.append(f1_score)

    # create final stats dict
    stats = {}
    stats["total_time"] = total_time
    stats["total_cost"] = total_cost
    stats["f1"] = np.mean(scores)
    with open(f'opt-profiling-data/mmqa-baseline-seed-{seed}-stats.json', 'w') as f:
        json.dump(stats, f)
    print(stats)
