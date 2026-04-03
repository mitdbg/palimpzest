"""
paper_run.py
============
Exhaustive evaluation of a 2-operator LLM pipeline on paper classification.

Logical plan:
    TextFile  →  summarizer  →  PaperFile  →  classifier  →  PaperClassification

Operator implementations per slot:
    LLMConvertBonded:  model ∈ {S, M, W}                                          →  3 configs
    CritiqueAndRefine: (base, critic, refine) ∈ {S, W}³                           →  8 configs
    MixtureOfAgents:   agents=[a,b] distinct from {S,M,W}, aggregator ∈ {a, b}    →  6 configs
    ─────────────────────────────────────────────────────────────────────────────────────────────
    Total per slot: 17   ×   Total plans: 17 × 17 = 289

Ground-truth labels (has_f1):
    paper1 → True,  paper2 → False,  paper3 → True
"""
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


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PaperFile(BaseModel):
    summary: str = Field(description="The summary of the research paper.")


class PaperClassification(BaseModel):
    has_f1: bool = Field(
        description="'True' if the research paper uses f1 score to measure performance, 'False' otherwise"
    )


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR = os.path.join(REPO_ROOT, "testdata", "paper")

PAPER_IDS = ["1", "2", "3"]
# Ground-truth: papers 1 and 3 report f1 score; paper 2 does not
HAS_F1_GROUND_TRUTH = {"1": True, "2": False, "3": True}

paper_dataset: list[tuple[str, DataRecord]] = []
for idx in PAPER_IDS:
    file_path = os.path.join(PAPER_DIR, f"paper{idx}.txt")
    with open(file_path, "r") as f:
        contents = f.read()
    record = DataRecord(
        data_item=TextFile(filename=f"paper{idx}.txt", contents=contents),
        source_indices=idx,
    )
    paper_dataset.append((idx, record))

print(f"Loaded {len(paper_dataset)} papers.")


# ---------------------------------------------------------------------------
# Model tiers
# ---------------------------------------------------------------------------

STRONG = Model.GPT_5_MINI
MEDIUM = Model.GPT_5_NANO
WEAK   = Model.GPT_4_1_NANO

MOA_TEMPS = [0.7, 0.7]


# ---------------------------------------------------------------------------
# Build OP_CONFIGS
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
# Helper: run one physical plan over all papers
# ---------------------------------------------------------------------------

def run_plan(op_summarizer, op_classifier, plan_label, combo_idx, n_combos):
    print(
        f"\n[{combo_idx}/{n_combos}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {plan_label}",
        flush=True,
    )
    try:
        total_latency = 0.0
        total_cost = 0.0
        correct = 0

        for paper_id, record in paper_dataset:
            print(f"  paper{paper_id}", end=" ", flush=True)
            gt_has_f1 = HAS_F1_GROUND_TRUTH[paper_id]

            drs_0 = op_summarizer(record)
            rec_0 = drs_0.data_records[0]
            stats_0 = drs_0.record_op_stats[0]
            summary = rec_0.summary

            summary_record = DataRecord(
                data_item=PaperFile(summary=summary),
                source_indices=paper_id,
            )

            drs_1 = op_classifier(summary_record)
            rec_1 = drs_1.data_records[0]
            stats_1 = drs_1.record_op_stats[0]
            has_f1 = rec_1.has_f1

            print(f"has_f1={has_f1} (gt={gt_has_f1})", flush=True)

            total_latency += stats_0.llm_call_duration_secs + stats_1.llm_call_duration_secs
            total_cost += (
                stats_0.total_input_cost + stats_0.total_output_cost
                + stats_1.total_input_cost + stats_1.total_output_cost
            )
            if has_f1 == gt_has_f1:
                correct += 1

        quality = correct / len(paper_dataset)
        print(
            f"  quality={quality:.4f}  llm_latency={total_latency:.2f}s  "
            f"cost=${total_cost:.6f}  correct={correct}/{len(paper_dataset)}",
            flush=True,
        )
        return {
            "plan_label": plan_label,
            "quality": quality,
            "total_latency_secs": total_latency,
            "total_cost_usd": total_cost,
            "correct": correct,
            "n_papers": len(paper_dataset),
        }

    except Exception as exc:
        print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)
        return {
            "plan_label": plan_label,
            "quality": None,
            "total_latency_secs": None,
            "total_cost_usd": None,
            "correct": None,
            "n_papers": len(paper_dataset),
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Main loop: 17 summarizer configs × 17 classifier configs = 289 plans
# ---------------------------------------------------------------------------

all_results = []
N_TOTAL = len(OP_CONFIGS) ** 2  # 289

for global_idx, (sum_cfg, cls_cfg) in enumerate(iproduct(OP_CONFIGS, OP_CONFIGS), start=1):
    label = f"summarizer={sum_cfg['label']} | classifier={cls_cfg['label']}"

    op_summarizer = make_op(sum_cfg, TextFile,  PaperFile,           "paper_summarizer")
    op_classifier = make_op(cls_cfg, PaperFile, PaperClassification, "paper_classifier", depends_on=["summary"])

    result = run_plan(op_summarizer, op_classifier, label, global_idx, N_TOTAL)
    result["summarizer_impl"] = sum_cfg["label"]
    result["classifier_impl"] = cls_cfg["label"]
    all_results.append(result)

    # Intermediate save every 10 completed plans
    if global_idx % 10 == 0:
        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"paper_results_{global_idx}.csv")
        pd.DataFrame(all_results).to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Final save
# ---------------------------------------------------------------------------

pd.DataFrame(all_results).to_csv("paper_results.csv", index=False)
