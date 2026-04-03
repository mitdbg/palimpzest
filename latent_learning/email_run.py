"""
email_run.py
============
Exhaustive evaluation of a 3-operator LLM pipeline on Enron email classification.

Logical plan:
    TextFile  →  ss_extracter  →  SenderSubjectFile  ─┬→  fraud_classifier     →  FraudClassification
                                                       └→  internal_classifier  →  internalClassification

Operator implementations per slot (same 17 as paper_run.py):
    LLMConvertBonded:  model ∈ {S, M, W}                                          →  3 configs
    CritiqueAndRefine: (base, critic, refine) ∈ {S, W}³                           →  8 configs
    MixtureOfAgents:   agents=[a,b] distinct from {S,M,W}, aggregator ∈ {a, b}    →  6 configs
    ─────────────────────────────────────────────────────────────────────────────────────────────
    Total per slot: 17   ×   Total plans: 17 × 17 = 289

Two experimental loops:
    1. full_loop   — all emails in testdata/enron-eval-medium/, 289 plans
    2. small_loop  — 4 hardcoded emails (1 labeled, 3 unlabeled),  289 plans

Scoring (per email):
    - labeled email:   correct iff is_fraud=True AND is_internal=True AND sender+subject match GT
    - unlabeled email: correct iff is_fraud=False AND is_internal=False
Quality = fraction of emails correctly scored.
"""
import glob
import json
import os
from datetime import datetime
from itertools import combinations, product as iproduct

import pandas as pd
from pydantic import BaseModel, Field

from palimpzest.constants import Model
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import TextFile
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.operators.critique_and_refine import CritiqueAndRefineConvert
from palimpzest.query.operators.mixture_of_agents import MixtureOfAgentsConvert

import litellm
import logging


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SenderSubjectFile(BaseModel):
    filename: str = Field(description="The UNIX-style name of the file")
    contents: str = Field(description="The contents of the file")
    sender: str = Field(description="The email address of the email's sender")
    subject: str = Field(description="The subject of the email")


class FraudClassification(BaseModel):
    is_fraud: bool = Field(
        description=(
            "'True' if the email refers to a fraudulent scheme "
            "(i.e., 'Raptor', 'Deathstar', 'Chewco', and/or 'Fat Boy'), 'False' otherwise"
        )
    )


class internalClassification(BaseModel):
    is_internal: bool = Field(
        description=(
            "'True' if the email is not quoting from a news article or an article "
            "written by someone outside of Enron, 'False' otherwise"
        )
    )


# ---------------------------------------------------------------------------
# Load full dataset and labels
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMAIL_DIR = os.path.join(REPO_ROOT, "testdata", "enron-eval-medium")
LABELS_FILE = os.path.join(REPO_ROOT, "enron-eval-medium-labels.json")

with open(LABELS_FILE, "r") as f:
    full_labels: dict[str, list] = json.load(f)

email_paths = sorted(glob.glob(os.path.join(EMAIL_DIR, "*.txt")))

full_dataset: list[tuple[str, DataRecord]] = []
for path in email_paths:
    filename = os.path.basename(path)
    with open(path, "r", errors="replace") as f:
        contents = f.read()
    record = DataRecord(
        data_item=TextFile(filename=filename, contents=contents),
        source_indices=filename,
    )
    full_dataset.append((filename, record))

n_labeled = sum(1 for fn, _ in full_dataset if full_labels.get(fn))
print(f"Loaded {len(full_dataset)} emails: {n_labeled} labeled, {len(full_dataset) - n_labeled} unlabeled.")


# ---------------------------------------------------------------------------
# Small 4-email dataset (1 labeled/fraudulent+internal, 3 unlabeled)
# ---------------------------------------------------------------------------

SMALL_FILES = [
    "delainey-d-sent-295.txt",
    "germany-c-inbox-19.txt",
    "donohoe-t-inbox-9.txt",
    "giron-d-inbox-9.txt",
]
SMALL_LABELS = {
    "delainey-d-sent-295.txt": [{"sender": "david.delainey@enron.com", "subject": "AIG Fund"}],
    "germany-c-inbox-19.txt": [],
    "donohoe-t-inbox-9.txt": [],
    "giron-d-inbox-9.txt": [],
}

small_dataset: list[tuple[str, DataRecord]] = []
for filename in SMALL_FILES:
    path = os.path.join(EMAIL_DIR, filename)
    with open(path, "r", errors="replace") as f:
        contents = f.read()
    record = DataRecord(
        data_item=TextFile(filename=filename, contents=contents),
        source_indices=filename,
    )
    small_dataset.append((filename, record))

print(f"Small dataset: {len(small_dataset)} emails.")


# ---------------------------------------------------------------------------
# Model tiers
# ---------------------------------------------------------------------------

STRONG = Model.GPT_5_MINI
MEDIUM = Model.GPT_5_NANO
WEAK   = Model.GPT_4_1_NANO

MOA_TEMPS = [0.7, 0.7]


# ---------------------------------------------------------------------------
# Build OP_CONFIGS (17 entries, same structure as paper_run.py)
# ---------------------------------------------------------------------------

OP_CONFIGS = []

# LLMConvertBonded: S, M, W  (3 configs)
for model in [STRONG, MEDIUM, WEAK]:
    OP_CONFIGS.append({
        "type": "LLM",
        "label": f"LCB({model.name})",
        "model": model,
    })

# CritiqueAndRefine: (base, critic, refine) ∈ {S, W}³  (2³ = 8 configs)
for base, critic, refine in iproduct([STRONG, WEAK], repeat=3):
    OP_CONFIGS.append({
        "type": "CAR",
        "label": f"CAR({base.name}, {critic.name}, {refine.name})",
        "model": base,
        "critic": critic,
        "refine": refine,
    })

# MixtureOfAgents: agents=[a, b] distinct from {S,M,W}, aggregator ∈ {a, b}  (3 pairs × 2 = 6 configs)
for agent_a, agent_b in combinations([STRONG, MEDIUM, WEAK], 2):
    for agg in [agent_a, agent_b]:
        OP_CONFIGS.append({
            "type": "MOA",
            "label": f"MOA(({agent_a.name}, {agent_b.name}), {agg.name})",
            "agents": [agent_a, agent_b],
            "agg": agg,
        })

assert len(OP_CONFIGS) == 17, f"Expected 17 configs, got {len(OP_CONFIGS)}"
print(f"Built {len(OP_CONFIGS)} operator configs  (3 LLM + 8 CAR + 6 MOA).")


# ---------------------------------------------------------------------------
# Helper: instantiate an operator from a config entry
# ---------------------------------------------------------------------------

def make_op(cfg, input_schema, output_schema, logical_op_id, depends_on=None):
    kwargs = dict(input_schema=input_schema, output_schema=output_schema, logical_op_id=logical_op_id)
    if depends_on:
        kwargs["depends_on"] = depends_on
    if cfg["type"] == "LLM":
        return LLMConvertBonded(model=cfg["model"], **kwargs)
    elif cfg["type"] == "CAR":
        return CritiqueAndRefineConvert(
            model=cfg["model"], critic_model=cfg["critic"], refine_model=cfg["refine"], **kwargs
        )
    else:  # MOA
        return MixtureOfAgentsConvert(
            proposer_models=cfg["agents"], temperatures=MOA_TEMPS, aggregator_model=cfg["agg"], **kwargs
        )


# ---------------------------------------------------------------------------
# Helper: run one physical plan over the given dataset
# ---------------------------------------------------------------------------

def run_plan(op_ss, op_fraud, op_internal, plan_label, combo_idx, n_combos, dataset, labels):
    print(
        f"\n[{combo_idx}/{n_combos}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {plan_label}",
        flush=True,
    )
    try:
        total_latency = 0.0
        total_cost = 0.0
        correct = 0

        print(f"  Running on {len(dataset)} emails...", flush=True)
        for i, (filename, record) in enumerate(dataset):
            print(f"  [{i+1}/{len(dataset)}] {filename}", end=" ", flush=True)
            gt_entries = labels.get(filename, [])
            gt_is_labeled = len(gt_entries) > 0

            drs_0 = op_ss(record)
            rec_0 = drs_0.data_records[0]
            stats_0 = drs_0.record_op_stats[0]
            sender = rec_0.sender or ""
            subject = rec_0.subject or ""

            ss_record = DataRecord(
                data_item=SenderSubjectFile(
                    filename=filename,
                    contents=record.contents,
                    sender=sender,
                    subject=subject,
                ),
                source_indices=filename,
            )

            drs_1 = op_fraud(ss_record)
            rec_1 = drs_1.data_records[0]
            stats_1 = drs_1.record_op_stats[0]
            is_fraud = rec_1.is_fraud

            drs_2 = op_internal(ss_record)
            rec_2 = drs_2.data_records[0]
            stats_2 = drs_2.record_op_stats[0]
            is_internal = rec_2.is_internal
            print(f"sender={sender!r}, is_fraud={is_fraud}, is_internal={is_internal}", flush=True)

            total_latency += (
                stats_0.llm_call_duration_secs
                + stats_1.llm_call_duration_secs
                + stats_2.llm_call_duration_secs
            )
            total_cost += (
                stats_0.total_input_cost + stats_0.total_output_cost
                + stats_1.total_input_cost + stats_1.total_output_cost
                + stats_2.total_input_cost + stats_2.total_output_cost
            )

            if gt_is_labeled:
                gt_sender = gt_entries[0]["sender"]
                gt_subject = gt_entries[0]["subject"]
                sender_match = sender.strip().lower() == gt_sender.strip().lower()
                subject_match = subject.strip().lower() == gt_subject.strip().lower()
                if is_fraud and is_internal and sender_match and subject_match:
                    correct += 1
            else:
                if not is_fraud and is_internal:
                    correct += 1

        quality = correct / len(dataset)
        print(
            f"  quality={quality:.4f}  llm_latency={total_latency:.2f}s  "
            f"cost=${total_cost:.6f}  correct={correct}/{len(dataset)}",
            flush=True,
        )
        return {
            "plan_label": plan_label,
            "quality": quality,
            "total_latency_secs": total_latency,
            "total_cost_usd": total_cost,
            "correct": correct,
            "n_emails": len(dataset),
        }

    except Exception as exc:
        print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)
        return {
            "plan_label": plan_label,
            "quality": None,
            "total_latency_secs": None,
            "total_cost_usd": None,
            "correct": None,
            "n_emails": len(dataset),
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Loop 1: full dataset — 17 × 17 = 289 plans
# ---------------------------------------------------------------------------

N_TOTAL = len(OP_CONFIGS) ** 2  # 289
# full_results = []
# full_out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "email_results_full.csv")

# for global_idx, (ss_cfg, cls_cfg) in enumerate(iproduct(OP_CONFIGS, OP_CONFIGS), start=1):
#     label = f"ss={ss_cfg['label']} | classifiers={cls_cfg['label']}"
#     op_ss       = make_op(ss_cfg,  TextFile,          SenderSubjectFile,     "ss_extracter")
#     op_fraud    = make_op(cls_cfg, SenderSubjectFile,  FraudClassification,   "fraud_classifier",    depends_on=["sender", "subject", "contents"])
#     op_internal = make_op(cls_cfg, SenderSubjectFile,  internalClassification, "internal_classifier", depends_on=["sender", "subject", "contents"])

#     result = run_plan(op_ss, op_fraud, op_internal, label, global_idx, N_TOTAL, full_dataset, full_labels)
#     result["ss_impl"]         = ss_cfg["label"]
#     result["classifier_impl"] = cls_cfg["label"]
#     full_results.append(result)

#     if global_idx % 5 == 0:
#         pd.DataFrame(full_results).to_csv(full_out_path, index=False)
#         print(f"  [saved {global_idx} results → {full_out_path}]", flush=True)

# pd.DataFrame(full_results).to_csv(full_out_path, index=False)
# print(f"\nFull-dataset loop done. {len(full_results)} results saved to {full_out_path}")


# ---------------------------------------------------------------------------
# Loop 2: small 4-email dataset — 17 × 17 = 289 plans
# ---------------------------------------------------------------------------

small_results = []

for global_idx, (ss_cfg, cls_cfg) in enumerate(iproduct(OP_CONFIGS, OP_CONFIGS), start=1):
    label = f"ss={ss_cfg['label']} | classifiers={cls_cfg['label']}"
    op_ss       = make_op(ss_cfg,  TextFile,          SenderSubjectFile,     "ss_extracter")
    op_fraud    = make_op(cls_cfg, SenderSubjectFile,  FraudClassification,   "fraud_classifier",    depends_on=["sender", "subject", "contents"])
    op_internal = make_op(cls_cfg, SenderSubjectFile,  internalClassification, "internal_classifier", depends_on=["sender", "subject", "contents"])

    result = run_plan(op_ss, op_fraud, op_internal, label, global_idx, N_TOTAL, small_dataset, SMALL_LABELS)
    result["ss_impl"]         = ss_cfg["label"]
    result["classifier_impl"] = cls_cfg["label"]
    small_results.append(result)

    if global_idx % 5 == 0:
        small_out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"email_results_small_{global_idx}.csv")
        pd.DataFrame(small_results).to_csv(small_out_path, index=False)

pd.DataFrame(small_results).to_csv("email_results_small.csv", index=False)
