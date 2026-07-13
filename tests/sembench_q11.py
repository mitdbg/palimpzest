#!/usr/bin/env python3
"""Run explicit Palimpzest Q11 stages for inspecting empty outputs."""

from __future__ import annotations
import litellm

litellm.suppress_debug_info = True

import json
import sys

import pandas as pd
from sklearn.metrics import f1_score
import palimpzest as pz
from palimpzest.core.lib.schemas import AudioFilepath, ImageFilepath

sys.path.append(".")


PZ_K = 3
PZ_BUDGET = 3
MAX_WORKERS = 8

CAR_ID = {"name": "car_id", "type": int, "desc": "The integer id for the car"}
COLS = [
    {"name": "transmission", "type": str, "desc": "The string transmission type"},
    {"name": "fuel_type", "type": str, "desc": "The string fuel type"},
    {"name": "vin", "type": str, "desc": "The vehicle identification number"},
    {"name": "image_id", "type": int, "desc": "The integer id for the car image"},
    {
        "name": "image_path",
        "type": ImageFilepath,
        "desc": "The filepath containing the car image",
    },
    {
        "name": "complaint_id",
        "type": int,
        "desc": "The integer id for the complaint text",
    },
    {
        "name": "summary",
        "type": str,
        "desc": "The string summary of the complaint text",
    },
    {"name": "year", "type": int, "desc": "The integer year of the car"},
    {"name": "mileage", "type": str, "desc": "The string make of the car"},
    {"name": "registration_date", "type": str, "desc": "The string model of the car"},
    {"name": "country", "type": str, "desc": "The string trim of the car"},
    {"name": "number_plate", "type": str, "desc": "The string body type of the car"},
]


class Q11Dataset(pz.IterDataset):
    def __init__(self, id: str, car_df: pd.DataFrame):
        super().__init__(id=id, schema=[CAR_ID, *COLS])
        self.car_df = car_df

    def __len__(self):
        return len(self.car_df)

    def __getitem__(self, idx: int):
        return self.car_df.iloc[idx].to_dict()


sf = "0.05"
query = 11
workload_path = f"/home/gerardo/semantic_htap/data/sembench/workload/cars_sf{sf}.json"

with open(workload_path, encoding="utf-8") as handle:
    workload = json.load(handle)

q_id = f"cars-sf{sf}-q{query}"
task = [t for t in workload if t["id"] == q_id][0]

dataset_path = f"~/semantic_htap/data/sembench/cars/sf{sf}/q{query}/"
records_path = f"{dataset_path}/records.csv"
images_path = f"{dataset_path}/images.csv"
audio_path = f"{dataset_path}/audio.csv"
complaints_path = f"{dataset_path}/complaints.json"

cars = pd.read_csv(records_path).set_index("car_id")
images = pd.read_csv(images_path).set_index("car_id")
complaints = pd.read_json(complaints_path, orient="index").set_index("car_id")

cars = cars.join(images, how="inner")
cars = cars.join(complaints, how="inner")
# cars = cars.join(complaints, how="left")
cars["car_id"] = cars.index


# append '/home/gerardo/' before the image_path
cars["image_path"] = cars["image_path"].apply(
    lambda x: f"/home/gerardo/semantic_htap/{x}"
)

print("Task")
print(f"  id: {task['id']}")
print(f"  query: {task.get('query', '')}")
print(f"  expected_answer: {task.get('answer')}")
print()

dataset = Q11Dataset(id="q11-car-data", car_df=cars)
dataset = dataset.filter(lambda row: row["fuel_type"] == "Electric")
dataset = dataset.sem_filter(
    "You are given an image of a vehicle or its parts. Return true if car is damaged.",
    # "The picture shows a car.",
    depends_on=["image_path"],
)
dataset = dataset.sem_filter(
    "The textual complaint mentions that the car or parts of it were on fire, burned, or had fire-related damages.",
    depends_on=["summary"],
)

pz_config = pz.QueryProcessorConfig(
    policy=pz.MaxQuality(),
    # policy=pz.MinCost(),
    execution_strategy="parallel",
    max_workers=MAX_WORKERS,
    k=PZ_K,
    join_parallelism=MAX_WORKERS,
    verbose=False,
    progress=True,
    sample_budget=PZ_BUDGET,
)

validator = pz.Validator(model=pz.Model.GPT_5)
output = dataset.optimize_and_run(validator=validator, config=pz_config)
result_df = output.to_df()

stats = output.execution_stats
if len(result_df):
    predicted = result_df["vin"].tolist()

    expected = task.get("answer")

    true_positives = len(set(predicted) & set(expected))
    false_positives = len(set(predicted) - set(expected))
    false_negatives = len(set(expected) - set(predicted))
    p = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    r = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    print(result_df["summary"].tolist())

else:
    predicted = []
    f1 = 0

print(result_df)
print("Predicted answer", predicted)
print("F1 score", f1)
print(f"  total_execution_cost: {stats.total_execution_cost}")
print(f"  optimization_time: {stats.optimization_time}")
print(f"  plan_execution_time: {stats.plan_execution_time}")
print(
    "  input_tokens: "
    f"{stats.input_text_tokens + stats.input_audio_tokens + stats.input_image_tokens}"
)
print(
    "  output_tokens: "
    f"{stats.cache_read_tokens + stats.cache_creation_tokens + stats.output_text_tokens + stats.embedding_input_tokens}"
)
print()

plan_stats = list(output.execution_stats.plan_stats.values())[0]
idx = 0
for unique_op_id, operator_stats in sorted(
    plan_stats.operator_stats.items(),
    key=lambda item: int(item[0].split("-", 1)[0]),
):
    model = None

    op_details = getattr(operator_stats, "op_details", None)
    if op_details:
        model = op_details.get("model")

    record_stats_list = getattr(operator_stats, "record_op_stats_lst", [])
    if not model:
        for record_stats in record_stats_list:
            model = getattr(record_stats, "model_name", None)
            if model:
                break

    print(f"{idx}. {operator_stats.op_name}: {model or 'no model'}")
    idx += 1
# plan = stats.plan_strs[list(stats.plan_strs.keys())[0]]
# print(plan)
