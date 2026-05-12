"""
legal_run.py
============
Exhaustive evaluation of a 3-operator LLM pipeline on CUAD TFC classification.

Logical plan:
    ESC: span_extractor -> summarizer -> classifier
    EC: span_extractor -> classifier

All operators: LLMConvertBonded with models from all_models (9 models).
300 plans randomly sampled from 9^3 = 729, run in parallel.

Evaluation per contract:
    - extractor_quality: F1 token match (normalized) vs. gold span. If no span
      produced (empty string), skip summary/classifier and auto-classify as no TFC.
    - summarizer_quality: GPT-5.4 judge — is summary sufficient to answer TFC
      question? NaN if the extractor produced no span.
    - classifier_quality: 1 if predicted is_TFC matches ground truth label, else 0.

Output: one CSV row per (plan, contract) with plan_label, per-op quality,
        produced_span, produced_summary.
"""

import random
import re
import string
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import product

import pandas as pd
from pydantic import BaseModel, Field
import litellm

from palimpzest.constants import Model
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.lib.schemas import TextFile
from palimpzest.query.operators.convert import LLMConvertBonded


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class Span(BaseModel):
    span: str | None = Field(default=None, description="""Highlight the parts (if any) of this contract related to "Termination For Convenience"
                that should be reviewed by a lawyer. Details: Can a party terminate this contract without cause
                (solely by giving a notice and allowing a waiting period to expire)? If no such part exists,
                return an empty string.""")

class Summary(BaseModel):
    summary: str | None = Field(default=None, description="""A concise summary of the contract's provisions
                         related to Termination For Convenience""")

class is_TFC(BaseModel):
    is_TFC: bool = Field(description="""Given the summary, return 'True' or 'False' to indicate
                         whether the contract contains a Termination For Convenience provision.""")


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join("testdata", "CUAD_tfc_10dataset.csv")

df_cuad = pd.read_csv(DATASET_PATH)
print(f"Loaded {len(df_cuad)} contracts.")

# list of (contract_idx, DataRecord, {label, span})
cuad_dataset: list[tuple[int, DataRecord, dict]] = []
for _, row in df_cuad.iterrows():
    gt_span = "" if pd.isna(row["span"]) else str(row["span"])
    record = DataRecord(
        data_item=TextFile(
            filename=f"contract_{row['contract_idx']}",
            contents=row["context"],
        ),
        source_indices=str(row["contract_idx"]),
    )
    cuad_dataset.append((int(row["contract_idx"]), record, {"label": int(row["label"]), "span": gt_span}))


# ---------------------------------------------------------------------------
# Model tiers
# ---------------------------------------------------------------------------

all_models = [
    Model.GPT_5,
    Model.GPT_5_MINI,
    Model.o4_MINI,
    Model.GPT_4_1,
    Model.GPT_5_NANO,
    Model.GPT_4_1_MINI,
    Model.GPT_4o,
    Model.GPT_4o_MINI,
    Model.GPT_4_1_NANO,
]  # decreasing mmlupro_overall score

EVAL_MODEL = Model.GPT_5_4  # for summary sufficiency evaluation


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    return ' '.join(s.split())


def f1_token_match(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def eval_summary_sufficient(summary: str) -> int:
    """Returns 1 if the eval model judges the summary sufficient to answer the TFC question, else 0."""
    prompt = (
        "You are evaluating whether a contract summary is sufficient to determine if the contract "
        "contains a 'Termination For Convenience' clause (i.e., whether a party can terminate the "
        "contract without cause by giving notice and allowing a waiting period to expire).\n\n"
        f"Summary: {summary}\n\n"
        "Is this summary sufficient to answer the TFC classification question? "
        "Answer with only 'Yes' or 'No'."
    )
    response = litellm.completion(
        model=EVAL_MODEL.value,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content.strip().lower()
    return 1 if answer.startswith("yes") else 0


# ---------------------------------------------------------------------------
# Run one plan
# ---------------------------------------------------------------------------

def make_lcb(model, input_schema, output_schema, logical_op_id):
    return LLMConvertBonded(
        model=model,
        input_schema=input_schema,
        output_schema=output_schema,
        logical_op_id=logical_op_id,
    )

def run_legal_plan_ec(op_extractor, op_classifier, plan_label, combo_idx, n_combos):
    lines = [f"\n[{combo_idx}/{n_combos}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {plan_label}"]
    rows = []
    for contract_idx, record, gt in cuad_dataset:
        gt_label = bool(gt["label"])
        gt_span = gt["span"]
        try:
            # --- Extractor ---
            ext_drs = op_extractor(record)
            produced_span = ext_drs.data_records[0].span or ""
            extractor_quality = f1_token_match(produced_span, gt_span)

            no_span = produced_span.strip() == ""

            # --- Classifier ---
            span_record = DataRecord(
                data_item=Span(span=produced_span),
                source_indices=str(contract_idx),
            )
            cls_drs = op_classifier(span_record)
            predicted_label = cls_drs.data_records[0].is_TFC

            classifier_quality = int(predicted_label == gt_label)
            lines.append(
                f"  contract={contract_idx:<5} "
                f"span_f1={extractor_quality:.3f} cls_q={classifier_quality}"
            )

            rows.append({
                "plan_label": str(plan_label),
                "contract_idx": contract_idx,
                "extractor_quality": extractor_quality,
                "classifier_quality": classifier_quality,
                "produced_span": produced_span,
                "gt_label": int(gt_label),
                "gt_span": gt_span,
            })

        except Exception as exc:
            lines.append(f"  ERROR contract={contract_idx}: {type(exc).__name__}: {exc}")
            rows.append({
                "plan_label": str(plan_label),
                "contract_idx": contract_idx,
                "extractor_quality": None,
                "classifier_quality": None,
                "produced_span": None,
                "gt_label": int(gt_label),
                "gt_span": gt_span,
                "error": str(exc),
            })

    return rows, "\n".join(lines)

def run_legal_plan_esc(op_extractor, op_summarizer, op_classifier, plan_label, combo_idx, n_combos):
    lines = [f"\n[{combo_idx}/{n_combos}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {plan_label}"]
    rows = []
    for contract_idx, record, gt in cuad_dataset:
        gt_label = bool(gt["label"])
        gt_span = gt["span"]
        try:
            # --- Extractor ---
            ext_drs = op_extractor(record)
            produced_span = ext_drs.data_records[0].span or ""
            extractor_quality = f1_token_match(produced_span, gt_span)

            no_span = produced_span.strip() == ""

            # --- Summarizer (skip if no span produced) ---
            if no_span:
                produced_summary = ""
                summarizer_quality = float("nan")
                predicted_label = False
            else:
                span_record = DataRecord(
                    data_item=Span(span=produced_span),
                    source_indices=str(contract_idx),
                )
                sum_drs = op_summarizer(span_record)
                produced_summary = sum_drs.data_records[0].summary or ""
                summarizer_quality = eval_summary_sufficient(produced_summary)

                # --- Classifier ---
                summary_record = DataRecord(
                    data_item=Summary(summary=produced_summary),
                    source_indices=str(contract_idx),
                )
                cls_drs = op_classifier(summary_record)
                predicted_label = cls_drs.data_records[0].is_TFC

            classifier_quality = int(predicted_label == gt_label)
            lines.append(
                f"  contract={contract_idx:<5} "
                f"span_f1={extractor_quality:.3f}  sum_q={summarizer_quality}  cls_q={classifier_quality}"
            )

            rows.append({
                "plan_label": str(plan_label),
                "contract_idx": contract_idx,
                "extractor_quality": extractor_quality,
                "summarizer_quality": summarizer_quality,
                "classifier_quality": classifier_quality,
                "produced_span": produced_span,
                "produced_summary": produced_summary,
                "gt_label": int(gt_label),
                "gt_span": gt_span,
            })

        except Exception as exc:
            lines.append(f"  ERROR contract={contract_idx}: {type(exc).__name__}: {exc}")
            rows.append({
                "plan_label": str(plan_label),
                "contract_idx": contract_idx,
                "extractor_quality": None,
                "summarizer_quality": None,
                "classifier_quality": None,
                "produced_span": None,
                "produced_summary": None,
                "gt_label": int(gt_label),
                "gt_span": gt_span,
                "error": str(exc),
            })

    return rows, "\n".join(lines)


# ---------------------------------------------------------------------------
# Main loop: 400 randomly sampled plans from 9^3 = 729, run in parallel
# ---------------------------------------------------------------------------

N_SAMPLE   = 400
N_WORKERS  = 10
SAVE_EVERY = 10
SEED       = 42
OUT_PATH   = "legal_results_ec.csv"

random.seed(SEED)
all_plans = list(product(all_models, repeat=2))
# sampled_plans = random.sample(all_plans, N_SAMPLE)
sampled_plans = all_plans
n_combos = len(sampled_plans)
print(f"Running {n_combos} sampled plans (seed={SEED}) over {len(cuad_dataset)} contracts.")


def run_plan_task(combo_idx: int, plan: tuple) -> tuple[list[dict], str]:
    model_ext, model_cls = plan
    plan_label = (model_ext.name, model_cls.name)
    op_extractor  = make_lcb(model_ext, TextFile, Span, "tfc_extractor")
    op_classifier = make_lcb(model_cls, Span, is_TFC, "tfc_classifier")
    return run_legal_plan_ec(op_extractor, op_classifier, plan_label, combo_idx, n_combos)


all_results = []
with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {
        executor.submit(run_plan_task, i + 1, plan): i
        for i, plan in enumerate(sampled_plans)
    }
    completed = 0
    for future in as_completed(futures):
        rows, output = future.result()
        print(output, flush=True)
        all_results.extend(rows)
        completed += 1
        if completed % SAVE_EVERY == 0:
            pd.DataFrame(all_results).to_csv(OUT_PATH, index=False)
            print(f"  [checkpoint: {completed}/{n_combos} plans saved -> {OUT_PATH}]", flush=True)

pd.DataFrame(all_results).to_csv(OUT_PATH, index=False)
print(f"Done. {len(all_results)} rows saved to {OUT_PATH}")
