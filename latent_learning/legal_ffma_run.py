"""
legal_ffma_run.py
=================
Full experimental loop: filter-filter-map/aggregate pipelines on testdata/100_CUAD.csv.

Logical plan dimensions:
    - 8 filter pairs: 4C2=6 from Group1 provisions + 2 specific pairs
    - 2 extract fields: Governing Law, Expiration Date
    - 2 plan types: FFM (filter-filter-map), FFA (filter-filter-aggregate)
    - 2 filter orderings per pair (original and swapped)

Physical plan: 3^3 = 27 model combos per logical plan.
Total: 8 * 2 * 3 * 27 = 1296 physical plans.

Intermediate caching: filter results are cached in 100_CUAD_intermediate.csv so repeated
filter steps across plans are not re-executed. Main results only record map/agg cost.
"""

import json
import os
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from itertools import combinations, product

import pandas as pd
from pydantic import Field, create_model

from palimpzest.constants import Model, PromptStrategy
from palimpzest.core.elements.filters import Filter
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import TextFile
from palimpzest.query.operators.aggregate import SemanticAggregate
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.operators.filter import LLMFilter


# ---------------------------------------------------------------------------
# Configuration — fill in models before running
# ---------------------------------------------------------------------------

DATASET_PATH = "testdata/100_CUAD.csv"
OUT_PATH = "latent_learning/legal_ffma_results.csv"
INTERMEDIATE_PATH = "latent_learning/100_CUAD_intermediate.csv"
QUALITY_PATH = "latent_learning/legal_ffma_quality_results.csv"
N_WORKERS = 10
SAVE_EVERY = 10

MODELS = [
    Model.GPT_5_MINI,
    Model.GPT_4_1,
    Model.GPT_4o_MINI,
]
AGG_MODELS = [
    Model.GPT_4_1,
    Model.GPT_4_1_MINI,
    Model.GPT_4_1_NANO,
]

# ---------------------------------------------------------------------------
# Provision and field info
# ---------------------------------------------------------------------------

PROVISION_INFO = {
    "Anti-Assignment": {
        "description": "Is consent or notice required of a party if the contract is assigned to a third party?",
    },
    "License Grant": {
        "description": "Does the contract contain a license granted by one party to its counterparty?",
    },
    "Cap on Liability": {
        "description": "Does the contract include a cap on liability upon the breach of a party's obligation? This includes time limitation for the counterparty to bring claims or maximum amount for recovery.",
    },
    "Audit Rights": {
        "description": "Does a party have the right to audit the books, records, or physical locations of the counterparty to ensure compliance with the contract?",
    },
    "Revenue/Profit Sharing": {
        "description": "Is one party required to share revenue or profit with the counterparty for any technology, goods, or services?",
    },
    "Post-Termination Services": {
        "description": "Is a party subject to obligations after the termination or expiration of a contract, including any post-termination transition, payment, transfer of IP, wind-down, last-buy, or similar commitments?",
    },
    "Insurance": {
        "description": "Is there a requirement for insurance that must be maintained by one party for the benefit of the counterparty?",
    },
    "Minimum Commitment": {
        "description": "Is there a minimum order size or minimum amount or units per-time period that one party must buy from the counterparty under the contract?",
    },
    "Exclusivity": {
        "description": (
            'Is there an exclusive dealing commitment with the counterparty? This includes a commitment to procure '
            'all "requirements" from one party of certain technology, goods, or services or a prohibition on '
            'licensing or selling technology, goods or services to third parties, or a prohibition on collaborating '
            'or working with other parties, whether during the contract or after the contract ends (or both).'
        ),
    },
}

FIELD_INFO = {
    "Governing Law": {
        "description": "Which state/country's law governs the interpretation of the contract?",
        "format": "Name of a US State, non-US Province, or Country. Do not include auxiliary phrases such as 'State of'",
    },
    "Expiration Date": {
        "description": "On what date will the contract's initial term expire?",
        "format": "Date (mm/dd/yy) / Perpetual. Use brackets '[]' for missing date information",
    },
    "Agreement Date": {
        "description": "The date of the contract",
        "format": "Date (mm/dd/yyyy). Use brackets '[]' for missing date information",
    },
}

FIELD_GT_COL = {
    "Governing Law":    "Governing Law-Answer",
    "Expiration Date":  "Expiration Date-Answer",
    "Agreement Date":   "Agreement Date-Answer",
}


# ---------------------------------------------------------------------------
# Filter pairs
# ---------------------------------------------------------------------------

TRAIN_GROUP_PROVISIONS = [
    "Anti-Assignment",
    "License Grant",
    "Cap on Liability",
    "Audit Rights",

    # "Revenue/Profit Sharing",
    # "Post-Termination Services",
]

GROUP1_PAIRS = list(combinations(TRAIN_GROUP_PROVISIONS, 2))  # 15 pairs

TEST_PAIRS = [
    # ("Anti-Assignment", "Insurance"),
    # ("License Grant", "Insurance"),
    # ("Minimum Commitment", "Insurance"),
    ("Exclusivity", "Cap on Liability"),
    ("Exclusivity", "Insurance"),
]


ALL_FILTER_PAIRS = GROUP1_PAIRS + TEST_PAIRS  # 8 pairs
MAP_FIELDS = ['Governing Law', 'Expiration Date']
AGG_FIELDS = ['Governing Law',]
PLAN_TYPES = ["FFM", "FFA"]
FILTER_ORDERS = [0, 1]


# ---------------------------------------------------------------------------
# Schema builders
# ---------------------------------------------------------------------------

def make_filter_condition(provision: str) -> str:
    info = PROVISION_INFO[provision]
    return (
        f"This contract contains a {provision} provision. "
        f"Description of provision: {info['description']}"
    )


def make_map_schema(field: str):
    info = FIELD_INFO[field]
    desc = (
        f"Extract the {field} value from this contract. "
        f"Description: {info['description']} "
        f"Answer format: {info['format']}"
    )
    safe_name = field.replace(" ", "_").replace("/", "_")
    Schema = create_model(
        f"MapSchema_{safe_name}",
        field_value=(str | None, Field(default=None, description=desc)),
    )
    return Schema, "field_value"


def make_agg_schema(field: str):
    info = FIELD_INFO[field]
    agg_str = (
        f"Extract and aggregate the {field} across all contracts into a deduplicated set "
        f"of unique values. Description: {info['description']}. Answer format: {info['format']}"
    )
    safe_name = field.replace(" ", "_")
    Schema = create_model(
        f"AggSchema_{safe_name}",
        unique_values=(list[str] | None, Field(
            default=None,
            description=f"Deduplicated set of {field} values found across all contracts.",
        )),
    )
    return Schema, "unique_values", agg_str


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

df = pd.read_csv(DATASET_PATH, index_col="idx")
dataset: list[tuple[int, str, DataRecord]] = []
for idx, row in df.iterrows():
    record = DataRecord(
        data_item=TextFile(
            filename=str(row["Filename"]),
            contents=str(row["contract_context"]),
        ),
        source_indices=str(idx),
    )
    dataset.append((int(idx), str(row["Filename"]), record))

idx_to_record: dict[int, DataRecord] = {idx: record for idx, _, record in dataset}
all_idx = sorted(df.index.tolist())
print(f"Loaded {len(dataset)} contracts.")


# ---------------------------------------------------------------------------
# GT quality helpers
# ---------------------------------------------------------------------------

_NORM_PREFIXES = ["State of ", "Commonwealth of ", "Province of ", "Territory of "]
_NORM_SUFFIXES = [
    ", United States of America",
    ", United States",
    ", U.S.A.",
    ", USA",
    " (USA)",
    " USA",
    ", U.S.",
    ", US",
]


def _normalize(s, field) -> str | None:
    if not s or str(s).lower() in ("nan", "none", ""):
        return None
    s = str(s).strip()

    if field == "Expiration Date":
        if s.lower() == "perpetual":
            return "Perpetual"
        m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", s)
        if m:
            month = m.group(1).zfill(2)
            day   = m.group(2).zfill(2)
            year  = m.group(3)
            year = year[-2:]
            return f"{month}/{day}/{year}"
        return s or None

    elif field == "Governing Law":
        for p in _NORM_PREFIXES:
            if s.lower().startswith(p.lower()):
                s = s[len(p):]
                break
        for sfx in _NORM_SUFFIXES:
            if s.lower().endswith(sfx.lower()):
                s = s[: -len(sfx)]
                break
        if s.lower() == "china":
            return "people's republic of china"
        return s.strip() or None


def _has_provision(val) -> bool:
    if pd.isna(val):
        return False
    return str(val).strip() not in ("", "[]")


def _iou_multiset(predicted: list | None, gt_list: list, field: str = "Governing Law") -> float:
    """Multiset Jaccard: normalize both lists (keep duplicates), sum(min)/sum(max)."""
    norm_gt = [_normalize(v, field).lower() for v in gt_list if _normalize(v, field)]
    if not predicted:
        return 0.0
    norm_pred = [_normalize(v, field).lower() for v in predicted if _normalize(v, field)]
    pred_c = Counter(norm_pred)
    gt_c = Counter(norm_gt)
    all_keys = set(pred_c) | set(gt_c)
    intersection = sum(min(pred_c[k], gt_c[k]) for k in all_keys)
    union = sum(max(pred_c[k], gt_c[k]) for k in all_keys)
    return round(intersection / union, 3) if union > 0 else 0.0


@lru_cache(maxsize=None)
def _gt_passed_for_pair(p1: str, p2: str) -> frozenset:
    if not p1 or not p2:
        return frozenset()
    return frozenset(
        idx
        for idx in all_idx
        if _has_provision(df.loc[idx, p1]) and _has_provision(df.loc[idx, p2])
    )


def _compute_filter_f1(actual_passed: set, gt_passed: frozenset) -> float:
    tp = len(actual_passed & gt_passed)
    fp = len(actual_passed - gt_passed)
    fn = len(gt_passed - actual_passed)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(f1, 3)


def _compute_quality(
    plan_type: str, result, surviving_f2: list, p1: str, p2: str, field: str
) -> float | None:
    gt_passed = _gt_passed_for_pair(p1, p2)
    gt_col = FIELD_GT_COL.get(field)
    if not gt_col:
        return None

    if plan_type == "FFM":
        actual_passed = set(surviving_f2)
        union_size = len(actual_passed | gt_passed)
        if union_size == 0:
            return 0.0
        correct = 0
        for idx in actual_passed & gt_passed:
            pred = result.get(str(idx)) if isinstance(result, dict) else None
            raw_gt = df.loc[idx, gt_col]
            gt_val = str(raw_gt).strip() if not pd.isna(raw_gt) else None
            np_, ng_ = _normalize(pred, field), _normalize(gt_val, field)
            if np_ and ng_ and np_.lower() == ng_.lower():
                correct += 1
        return round(correct / union_size, 3)

    if plan_type == "FFA":
        gt_vals = [
            str(df.loc[idx, gt_col]).strip()
            for idx in gt_passed
            if not pd.isna(df.loc[idx, gt_col])
            and str(df.loc[idx, gt_col]).strip() not in ("", "nan")
        ]
        gt_unique = sorted(set(gt_vals))
        predicted = result if isinstance(result, list) else None
        return _iou_multiset(predicted, gt_unique, field)

    return None


# ---------------------------------------------------------------------------
# Intermediate filter cache
# ---------------------------------------------------------------------------

_inter_lock = threading.Lock()
_inter_cache: dict[tuple[str, str], dict] = {}

if os.path.exists(INTERMEDIATE_PATH):
    _df_inter = pd.read_csv(INTERMEDIATE_PATH)
    for _, _row in _df_inter.iterrows():
        _key = (_row["filter"], _row["filter_models"])
        _inter_cache[_key] = {
            "passed_idx": json.loads(_row["passed_idx"]),
            "filter_input_cost": float(_row["filter_input_cost"]),
            "filter_output_cost": float(_row["filter_output_cost"]),
            "filter_llm_call_duration_secs": float(_row["filter_llm_call_duration_secs"]),
        }
    print(f"Loaded {len(_inter_cache)} cached filter results from {INTERMEDIATE_PATH}.")


def _inter_key(provs: list[str], models: list[Model]) -> tuple[str, str]:
    return json.dumps(provs), json.dumps([m.name for m in models])


def _inter_lookup(provs: list[str], models: list[Model]) -> dict | None:
    key = _inter_key(provs, models)
    with _inter_lock:
        return _inter_cache.get(key)


def _inter_save(provs: list[str], models: list[Model], passed_idx: list[int],
                input_cost: float, output_cost: float, latency: float) -> None:
    key = _inter_key(provs, models)
    with _inter_lock:
        if key in _inter_cache:
            return  # already written by another thread
        _inter_cache[key] = {
            "passed_idx": passed_idx,
            "filter_input_cost": input_cost,
            "filter_output_cost": output_cost,
            "filter_llm_call_duration_secs": latency,
        }
        csv_row = {
            "filter": key[0],
            "filter_models": key[1],
            "passed_idx": json.dumps(passed_idx),
            "filter_input_cost": input_cost,
            "filter_output_cost": output_cost,
            "filter_llm_call_duration_secs": latency,
        }
        write_header = not os.path.exists(INTERMEDIATE_PATH)
        pd.DataFrame([csv_row]).to_csv(INTERMEDIATE_PATH, mode="a", header=write_header, index=False)


_quality_lock = threading.Lock()


def _quality_save(quality_row: dict) -> None:
    with _quality_lock:
        write_header = not os.path.exists(QUALITY_PATH)
        pd.DataFrame([quality_row]).to_csv(QUALITY_PATH, mode="a", header=write_header, index=False)


# ---------------------------------------------------------------------------
# Filter chain runners (used by pre-warm and run_physical_plan)
# ---------------------------------------------------------------------------

def _run_single_filter(prov: str, m: Model) -> None:
    """Run and cache a single filter over the full dataset. No-op if already cached."""
    if _inter_lookup([prov], [m]) is not None:
        return
    filter_op = LLMFilter(
        model=m,
        filter=Filter(filter_condition=make_filter_condition(prov)),
        input_schema=TextFile,
        output_schema=TextFile,
        logical_op_id=f"f1_{prov.replace(' ', '_')}",
    )
    surviving = []
    input_cost = output_cost = latency = 0.0
    for idx, _, record in dataset:
        drs = filter_op(record)
        for stat in drs.record_op_stats:
            input_cost += stat.total_input_cost
            output_cost += stat.total_output_cost
            latency += stat.llm_call_duration_secs
        if drs.data_records[0]._passed_operator:
            surviving.append(idx)
    _inter_save([prov], [m], surviving, input_cost, output_cost, latency)


def _run_two_filter(filt1_prov: str, filt2_prov: str, m1: Model, m2: Model) -> None:
    """Run and cache the two-filter chain. Assumes [filt1_prov, m1] is already cached."""
    if _inter_lookup([filt1_prov, filt2_prov], [m1, m2]) is not None:
        return
    surviving_f1 = _inter_lookup([filt1_prov], [m1])["passed_idx"]
    filter_op2 = LLMFilter(
        model=m2,
        filter=Filter(filter_condition=make_filter_condition(filt2_prov)),
        input_schema=TextFile,
        output_schema=TextFile,
        logical_op_id=f"f2_{filt2_prov.replace(' ', '_')}",
    )
    surviving_f2 = []
    f2_input_cost = f2_output_cost = f2_latency = 0.0
    for idx in surviving_f1:
        drs2 = filter_op2(idx_to_record[idx])
        for stat in drs2.record_op_stats:
            f2_input_cost += stat.total_input_cost
            f2_output_cost += stat.total_output_cost
            f2_latency += stat.llm_call_duration_secs
        if drs2.data_records[0]._passed_operator:
            surviving_f2.append(idx)
    f1_entry = _inter_lookup([filt1_prov], [m1])
    cum_input  = f1_entry["filter_input_cost"]              + f2_input_cost
    cum_output = f1_entry["filter_output_cost"]             + f2_output_cost
    cum_lat    = f1_entry["filter_llm_call_duration_secs"]  + f2_latency
    _inter_save([filt1_prov, filt2_prov], [m1, m2], surviving_f2, cum_input, cum_output, cum_lat)


def _run_filter_chain(filt1_prov: str, filt2_prov: str, m1: Model, m2: Model) -> None:
    """Ensure both single- and two-filter results are cached. No-op if already cached."""
    _run_single_filter(filt1_prov, m1)
    _run_two_filter(filt1_prov, filt2_prov, m1, m2)


# ---------------------------------------------------------------------------
# Run one physical plan (filters guaranteed cached by pre-warm)
# ---------------------------------------------------------------------------

def run_physical_plan(plan: dict) -> dict:
    p1 = plan["provision1"]
    p2 = plan["provision2"]
    field = plan["extract_field"]
    plan_type = plan["plan_type"]
    filter_order = plan["filter_order"]
    m1 = plan["model_filter1"]
    m2 = plan["model_filter2"]
    m3 = plan["model_map_agg"]

    filt1_prov, filt2_prov = (p1, p2) if filter_order == 0 else (p2, p1)

    _run_filter_chain(filt1_prov, filt2_prov, m1, m2)  # no-op if already cached
    surviving_f1 = _inter_lookup([filt1_prov], [m1])["passed_idx"]
    surviving_f2 = _inter_lookup([filt1_prov, filt2_prov], [m1, m2])["passed_idx"]

    # ---- Map or Aggregate (only these costs go to main results) ----
    survivors = [idx_to_record[idx] for idx in surviving_f2]
    result = None
    map_agg_input_cost = 0.0
    map_agg_output_cost = 0.0
    map_agg_latency = 0.0

    if plan_type == "FFM":
        MapSchema, map_field = make_map_schema(field)
        map_op = LLMConvertBonded(
            model=m3,
            input_schema=TextFile,
            output_schema=MapSchema,
            logical_op_id=f"map_{field.replace(' ', '_')}",
        )
        result_dict: dict = {}
        for idx, rec in zip(surviving_f2, survivors):
            drs = map_op(rec)
            for stat in drs.record_op_stats:
                map_agg_input_cost += stat.total_input_cost
                map_agg_output_cost += stat.total_output_cost
                map_agg_latency += stat.llm_call_duration_secs
            result_dict[idx] = getattr(drs.data_records[0], map_field, None)
        result = result_dict

    elif plan_type == "FFA":
        AggSchema, agg_field, agg_str = make_agg_schema(field)
        agg_op = SemanticAggregate(
            agg_str=agg_str,
            model=m3,
            prompt_strategy=PromptStrategy.AGG_NO_REASONING,
            input_schema=TextFile,
            output_schema=AggSchema,
            logical_op_id=f"agg_{field.replace(' ', '_')}",
        )
        if survivors:
            drs = agg_op(survivors)
            for stat in drs.record_op_stats:
                map_agg_input_cost += stat.total_input_cost
                map_agg_output_cost += stat.total_output_cost
                map_agg_latency += stat.llm_call_duration_secs
            if drs.data_records:
                result = getattr(drs.data_records[0], agg_field, None)
        else:
            result = []

    return {
        "query_plan": ((p1, p2, field), plan_type),
        "filter1_provision": filt1_prov,
        "filter2_provision": filt2_prov,
        "filter_order": filter_order,
        "extract_field": field,
        "plan_type": plan_type,
        "physical_plan": (m1.name, m2.name, m3.name),
        "model_filter1": m1.name,
        "model_filter2": m2.name,
        "model_map_agg": m3.name,
        "surviving_after_f1": json.dumps(surviving_f1),
        "surviving_after_f2": json.dumps(surviving_f2),
        "result": json.dumps(result),
        "MA_input_cost": map_agg_input_cost,
        "MA_output_cost": map_agg_output_cost,
        "MA_latency_secs": map_agg_latency,
    }


# ---------------------------------------------------------------------------
# Generate all plans and resume
# ---------------------------------------------------------------------------

all_plans = []
for pair in ALL_FILTER_PAIRS:
    p1, p2 = pair
    for filter_order in FILTER_ORDERS:
        for plan_type in PLAN_TYPES:
            fields = AGG_FIELDS if plan_type=='FFA' else MAP_FIELDS
            for field in fields:
                m3_pool = AGG_MODELS if plan_type == "FFA" else MODELS
                for m1, m2, m3 in product(MODELS, MODELS, m3_pool):
                    all_plans.append({
                        "provision1": p1,
                        "provision2": p2,
                        "filter_order": filter_order,
                        "extract_field": field,
                        "plan_type": plan_type,
                        "model_filter1": m1,
                        "model_filter2": m2,
                        "model_map_agg": m3,
                    })

print(f"Total physical plans: {len(all_plans)}")

existing_results: list[dict] = []
completed_keys: set[tuple] = set()
if os.path.exists(OUT_PATH):
    df_done = pd.read_csv(OUT_PATH)
    existing_results = df_done.to_dict("records")
    for _row in existing_results:
        completed_keys.add((
            _row["filter1_provision"], _row["filter2_provision"], int(_row["filter_order"]),
            _row["extract_field"], _row["plan_type"],
            _row["model_filter1"], _row["model_filter2"], _row["model_map_agg"],
        ))
    print(f"Resuming: {len(completed_keys)} plans already completed.")

remaining_plans = [
    p for p in all_plans
    if (
        (p["provision1"] if p["filter_order"] == 0 else p["provision2"]),
        (p["provision2"] if p["filter_order"] == 0 else p["provision1"]),
        p["filter_order"],
        p["extract_field"], p["plan_type"],
        p["model_filter1"].name, p["model_filter2"].name, p["model_map_agg"].name,
    ) not in completed_keys
]

print(f"Remaining plans to run: {len(remaining_plans)}")
all_results = list(existing_results)


# ---------------------------------------------------------------------------
# Pre-warm filter cache (two-phase parallel) — zero redundant LLM calls
# ---------------------------------------------------------------------------

# Collect unique single-filter and two-filter tasks from remaining plans
_single_tasks: set[tuple] = set()
_chain_tasks: set[tuple] = set()
for _p in remaining_plans:
    _filt1 = _p["provision1"] if _p["filter_order"] == 0 else _p["provision2"]
    _filt2 = _p["provision2"] if _p["filter_order"] == 0 else _p["provision1"]
    _m1, _m2 = _p["model_filter1"], _p["model_filter2"]
    _single_tasks.add((_filt1, _m1))
    _chain_tasks.add((_filt1, _filt2, _m1, _m2))

# Phase 1: single filters — all independent, run in parallel
_missing_singles = [(prov, m) for prov, m in _single_tasks if _inter_lookup([prov], [m]) is None]
print(f"Pre-warm phase 1: {len(_single_tasks)} unique single filters, {len(_missing_singles)} not yet cached.")
if _missing_singles:
    with ThreadPoolExecutor(max_workers=min(N_WORKERS, len(_missing_singles))) as _ex1:
        _futs1 = {_ex1.submit(_run_single_filter, prov, m): (prov, m) for prov, m in _missing_singles}
        for _i, _fut in enumerate(as_completed(_futs1), 1):
            _prov, _m = _futs1[_fut]
            _fut.result()
            print(f"  [phase1 {_i}/{len(_missing_singles)}] {_prov} m={_m.name}", flush=True)
print("Phase 1 complete.")

# Phase 2: two-filter chains — filt1 guaranteed cached, all independent
_missing_chains = [(f1, f2, m1, m2) for f1, f2, m1, m2 in _chain_tasks if _inter_lookup([f1, f2], [m1, m2]) is None]
print(f"Pre-warm phase 2: {len(_chain_tasks)} unique two-filter chains, {len(_missing_chains)} not yet cached.")
if _missing_chains:
    with ThreadPoolExecutor(max_workers=min(N_WORKERS, len(_missing_chains))) as _ex2:
        _futs2 = {_ex2.submit(_run_two_filter, f1, f2, m1, m2): (f1, f2, m1, m2) for f1, f2, m1, m2 in _missing_chains}
        for _i, _fut in enumerate(as_completed(_futs2), 1):
            _f1, _f2, _m1, _m2 = _futs2[_fut]
            _fut.result()
            print(f"  [phase2 {_i}/{len(_missing_chains)}] {_f1}+{_f2} m1={_m1.name} m2={_m2.name}", flush=True)
print("Phase 2 complete.")


# ---------------------------------------------------------------------------
# Execute with ThreadPoolExecutor
# ---------------------------------------------------------------------------

def run_plan_task(plan_idx: int, plan: dict) -> tuple[dict, str]:
    n = len(remaining_plans)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filt1 = plan["provision1"] if plan["filter_order"] == 0 else plan["provision2"]
    filt2 = plan["provision2"] if plan["filter_order"] == 0 else plan["provision1"]
    label = (
        f"[{plan_idx}/{n}] {ts} "
        f"{plan['plan_type']} {filt1}+{filt2} "
        f"field={plan['extract_field']} "
        f"models=({plan['model_filter1'].name},{plan['model_filter2'].name},{plan['model_map_agg'].name})"
    )
    try:
        row = run_physical_plan(plan)
        surviving_f2 = json.loads(row["surviving_after_f2"])
        nf1 = len(json.loads(row["surviving_after_f1"]))
        nf2 = len(surviving_f2)
        total_cost = row["MA_input_cost"] + row["MA_output_cost"]

        result_raw = json.loads(row["result"]) if row["result"] not in (None, "null") else None
        gt_passed = _gt_passed_for_pair(plan["provision1"], plan["provision2"])
        filter_f1 = _compute_filter_f1(set(surviving_f2), gt_passed)
        quality = _compute_quality(
            plan["plan_type"],
            result_raw,
            surviving_f2,
            plan["provision1"],
            plan["provision2"],
            plan["extract_field"],
        )
        _quality_save(
            {
                "query_plan": row["query_plan"],
                "filter_order": row["filter_order"],
                "physical_plan": row["physical_plan"],
                "quality": quality,
                "filter_f1": filter_f1,
            }
        )

        msg = f"{label} -> pass_f1={nf1} pass_f2={nf2} quality={quality} filter_f1={filter_f1} map_agg_cost=${total_cost:.4f} latency={row['MA_latency_secs']:.1f}s"
        return row, msg
    except Exception as exc:
        error_row = {
            "query_plan": ((plan["provision1"], plan["provision2"], plan["extract_field"]), plan["plan_type"]),
            "filter1_provision": filt1,
            "filter2_provision": filt2,
            "filter_order": plan["filter_order"],
            "extract_field": plan["extract_field"],
            "plan_type": plan["plan_type"],
            "physical_plan": (plan["model_filter1"].name, plan["model_filter2"].name, plan["model_map_agg"].name),
            "model_filter1": plan["model_filter1"].name,
            "model_filter2": plan["model_filter2"].name,
            "model_map_agg": plan["model_map_agg"].name,
            "surviving_after_f1": "[]",
            "surviving_after_f2": "[]",
            "result": "null",
            "MA_input_cost": 0.0,
            "MA_output_cost": 0.0,
            "MA_latency_secs": 0.0,
            "error": str(exc),
        }
        return error_row, f"{label} -> ERROR: {type(exc).__name__}: {exc}"


completed_count = 0
with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {
        executor.submit(run_plan_task, i + 1, plan): i
        for i, plan in enumerate(remaining_plans)
    }
    for future in as_completed(futures):
        row, msg = future.result()
        print(msg, flush=True)
        all_results.append(row)
        completed_count += 1
        if completed_count % SAVE_EVERY == 0:
            pd.DataFrame(all_results).to_csv(OUT_PATH, index=False)
            print(f"  [checkpoint: {completed_count}/{len(remaining_plans)} saved -> {OUT_PATH}]", flush=True)

pd.DataFrame(all_results).to_csv(OUT_PATH, index=False)
print(f"Done. {len(all_results)} total rows saved to {OUT_PATH}")
