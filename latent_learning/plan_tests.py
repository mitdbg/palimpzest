"""
plan_tests.py
=============
Exhaustive evaluation of a 3-operator LLM pipeline on Enron email classification.

Logical plan:
    TextFile  →  ss_extracter  →  SenderSubjectFile  ─┬→  fraud_classifier     →  FraudClassification
                                                       └→  internal_classifier  →  internalClassification

Domain: 9 GPT models per operator → 9³ = 729 physical plan combinations.

Scoring (per email):
    - labeled emails (non-empty label list): correct iff
          is_fraud=True AND is_internal=True
          AND predicted sender matches ground truth
          AND predicted subject matches ground truth
    - unlabeled emails (empty label list): correct iff is_fraud=False AND is_internal=False

Aggregated metrics (over all emails in the dataset):
    quality            = fraction of emails correctly scored
    total_latency_secs = sum of LLM call durations across all 3 operators over all emails
    total_cost_usd     = sum of (input + output) costs across all 3 operators over all emails
"""
import glob
import json
import os
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field

from palimpzest.constants import Model
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import TextFile
from palimpzest.query.operators.convert import LLMConvertBonded
from palimpzest.query.operators.critique_and_refine import CritiqueAndRefineConvert
from palimpzest.query.operators.mixture_of_agents import MixtureOfAgentsConvert

from itertools import product

import litellm
import logging


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class SenderSubjectFile(BaseModel):
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
# Load dataset and labels
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMAIL_DIR = os.path.join(REPO_ROOT, "testdata", "enron-eval-medium")
LABELS_FILE = os.path.join(REPO_ROOT, "enron-eval-medium-labels.json")

with open(LABELS_FILE, "r") as f:
    labels: dict[str, list] = json.load(f)

email_paths = sorted(glob.glob(os.path.join(EMAIL_DIR, "*.txt")))

email_dataset: list[tuple[str, DataRecord]] = []
for path in email_paths:
    filename = os.path.basename(path)
    with open(path, "r", errors="replace") as f:
        contents = f.read()
    record = DataRecord(
        data_item=TextFile(filename=filename, contents=contents),
        source_indices=filename,
    )
    email_dataset.append((filename, record))

n_labeled = sum(1 for fn, _ in email_dataset if labels.get(fn))
print(f"Loaded {len(email_dataset)} emails: {n_labeled} labeled, {len(email_dataset) - n_labeled} unlabeled.")

# ---------------------------------------------------------------------------
# Model tiers
# ---------------------------------------------------------------------------

STRONG = Model.GPT_5_MINI
MEDIUM = Model.GPT_5_NANO
WEAK   = Model.GPT_4_1_NANO


# ---------------------------------------------------------------------------
# Helper: run one physical plan (3 operators) over the full dataset
# ---------------------------------------------------------------------------

def run_plan(op_ss, op_fraud, op_internal, plan_label, combo_idx, n_combos):
    print(
        f"\n[{combo_idx}/{n_combos}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {plan_label}",
        flush=True,
    )
    try:
        total_latency = 0.0
        total_cost = 0.0
        correct = 0

        print(f"  Running on {len(email_dataset)} emails...", flush=True)
        for i, (filename, record) in enumerate(email_dataset):
            print(f"  [{i+1}/{len(email_dataset)}] {filename}", end=" ", flush=True)
            gt_entries = labels.get(filename, [])
            gt_is_labeled = len(gt_entries) > 0

            drs_0 = op_ss(record)
            rec_0 = drs_0.data_records[0]
            stats_0 = drs_0.record_op_stats[0]
            sender = rec_0.sender or ""
            subject = rec_0.subject or ""

            ss_record = DataRecord(
                data_item=SenderSubjectFile(sender=sender, subject=subject),
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
                if not is_fraud and not is_internal:
                    correct += 1

        quality = correct / len(email_dataset)
        print(
            f"  quality={quality:.4f}  "
            f"llm_latency={total_latency:.2f}s  cost=${total_cost:.6f}  "
            f"correct={correct}/{len(email_dataset)}",
            flush=True,
        )
        return {
            "plan_label": plan_label,
            "quality": quality,
            "total_latency_secs": total_latency,
            "total_cost_usd": total_cost,
            "correct": correct,
            "n_emails": len(email_dataset),
        }

    except Exception as exc:
        print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)
        return {
            "plan_label": plan_label,
            "quality": None,
            "total_latency_secs": None,
            "total_cost_usd": None,
            "correct": None,
            "n_emails": len(email_dataset),
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# 6 operator implementations (applied identically to each logical operator slot)
#
# Naming in the saved CSV:
#   LLMConvertBonded      → ss_impl / classifier_impl = "(base)"
#   CritiqueAndRefine     → "(base, critic, refine)"
#   MixtureOfAgents       → "((agent_1, agent_2), agg)"
# ---------------------------------------------------------------------------

MOA_TEMPS = [0.7, 0.7]

OP_CONFIGS = [
    # LLMConvertBonded: S, W
    {
        "type": "LLM",
        "label": f"({STRONG.name})",
        "model": STRONG,
    },
    {
        "type": "LLM",
        "label": f"({WEAK.name})",
        "model": WEAK,
    },
    # CritiqueAndRefine: (S,S,S), (W,W,W)
    {
        "type": "CAR",
        "label": f"({STRONG.name}, {STRONG.name}, {STRONG.name})",
        "model": STRONG, "critic": STRONG, "refine": STRONG,
    },
    {
        "type": "CAR",
        "label": f"({WEAK.name}, {WEAK.name}, {WEAK.name})",
        "model": WEAK, "critic": WEAK, "refine": WEAK,
    },
    # MixtureOfAgents: (S,W)→S, (S,W)→W
    {
        "type": "MOA",
        "label": f"(({STRONG.name}, {WEAK.name}), {STRONG.name})",
        "agents": [STRONG, WEAK], "agg": STRONG,
    },
    {
        "type": "MOA",
        "label": f"(({STRONG.name}, {WEAK.name}), {WEAK.name})",
        "agents": [STRONG, WEAK], "agg": WEAK,
    },
]


def make_op(cfg, input_schema, output_schema, logical_op_id, depends_on=None):
    """Instantiate a physical operator from an OP_CONFIGS entry."""
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
# Main loop: 6 ss configs × 6 classifier configs = 36 plans
# fraud_classifier and internal_classifier share the same implementation
# ---------------------------------------------------------------------------

all_results = []
N_TOTAL = 36

for global_idx, (ss_cfg, cls_cfg) in enumerate(product(OP_CONFIGS, OP_CONFIGS), start=1):
    label = f"ss={ss_cfg['label']} | classifiers={cls_cfg['label']}"
    op_ss       = make_op(ss_cfg,  TextFile,         SenderSubjectFile,    "ss_extracter")
    op_fraud    = make_op(cls_cfg, SenderSubjectFile, FraudClassification,  "fraud_classifier",    depends_on=["sender", "subject"])
    op_internal = make_op(cls_cfg, SenderSubjectFile, internalClassification, "internal_classifier", depends_on=["sender", "subject"])

    result = run_plan(op_ss, op_fraud, op_internal, label, global_idx, N_TOTAL)
    result["ss_impl"]         = ss_cfg["label"]
    result["classifier_impl"] = cls_cfg["label"]
    all_results.append(result)

    # Intermediate save every 3 completed plans
    if global_idx % 3 == 0:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"email_results_{global_idx}.csv")
        pd.DataFrame(all_results).to_csv(out_path, index=False)
        print(f"  [saved {global_idx} results → {out_path}]", flush=True)


# ---------------------------------------------------------------------------
# Final save
# ---------------------------------------------------------------------------

pd.DataFrame(all_results).to_csv(out_path, index=False)
print(f"\nDone. {len(all_results)} results saved to {out_path}")