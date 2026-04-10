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
from itertools import combinations, product

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

class Summary(BaseModel):
    summary: str = Field(description="The summary of the research paper.")


class is_F1(BaseModel):
    is_f1: bool = Field(
        description="'True' if the research paper uses f1 score to measure performance, 'False' otherwise"
    )


# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR = os.path.join(REPO_ROOT, "testdata", "paper")

PAPER_IDS = ["1", "2", "3"]
# Ground-truth: papers 1 and 3 report f1 score; paper 2 does not
HAS_F1_GROUND_TRUTH = {"paper1.txt": True, "paper2.txt": False, "paper3.txt": True}

paper_dataset: list[tuple[str, DataRecord]] = []
for idx in PAPER_IDS:
    file_path = os.path.join(PAPER_DIR, f"paper{idx}.txt")
    with open(file_path, "r") as f:
        contents = f.read()
    record = DataRecord(
        data_item=TextFile(filename=f"paper{idx}.txt", contents=contents),
        source_indices=idx,
    )
    paper_dataset.append((f"paper{idx}.txt", record))

print(f"Loaded {len(paper_dataset)} papers.")


# ---------------------------------------------------------------------------
# Model tiers
# ---------------------------------------------------------------------------
STRONG = Model.GPT_5_MINI
STRONG_MEDIUM = Model.GPT_4_1
MEDIUM = Model.GPT_5_NANO
WEAK_MEDIUM = Model.GPT_4o_MINI
WEAK   = Model.GPT_4_1_NANO
all_models = [Model.GPT_5,
        Model.GPT_5_MINI,
        Model.o4_MINI,
        Model.GPT_4_1,
        Model.GPT_5_NANO,
        Model.GPT_4_1_MINI,
        Model.GPT_4o,
        Model.GPT_4o_MINI,
        Model.GPT_4_1_NANO] #decreasing MMLU-Pro values

MOA_TEMPS = [0.7, 0.7]

# ---------------------------------------------------------------------------
# Build OP_CONFIGS
# ---------------------------------------------------------------------------

OP_CONFIGS = []

# LLMConvertBonded: S, M, W  (3 configs)
for model in [STRONG, MEDIUM, WEAK]:
    OP_CONFIGS.append({
        "type": "LCB",
        "label": f"LCB({model.name})",
        "model": model,
    })

# CritiqueAndRefine: (base, critic, refine) ∈ {S, W}³  (2³ = 8 configs)
for base, critic, refine in product([STRONG, WEAK], repeat=3):
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


def make_op(cfg, input_schema, output_schema, logical_op_id, depends_on=None):
    kwargs = dict(input_schema=input_schema, output_schema=output_schema, logical_op_id=logical_op_id)
    if depends_on:
        kwargs["depends_on"] = depends_on
    if cfg["type"] == "LCB":
        return LLMConvertBonded(model=cfg["model"], **kwargs)
    elif cfg["type"] == "CAR":
        return CritiqueAndRefineConvert(
            model=cfg["model"], critic_model=cfg["critic"], refine_model=cfg["refine"], **kwargs
        )
    elif cfg["type"] == "MOA":
        return MixtureOfAgentsConvert(
            proposer_models=cfg["agents"], temperatures=MOA_TEMPS, aggregator_model=cfg["agg"], **kwargs
        )
    else:
        raise ValueError(f"Unknown config type: {cfg['type']}")

def run_paper_plan(op_summary, op_f1, plan_label, combo_idx, n_combos):
    print(
        f"\n[{combo_idx}/{n_combos}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {plan_label}",
        flush=True,
    )
    try:
        op_latency = [0.0 for _ in range(2)]
        total_latency = 0.0
        op_cost = [0.0 for _ in range(2)]
        total_cost = 0.0
        correct = 0

        for i, (filename, record) in enumerate(paper_dataset):
            print(f"  [{i+1}/{len(paper_dataset)}] {filename}", end=" ", flush=True)
            gt_is_f1 = HAS_F1_GROUND_TRUTH[filename]

            summary_drs = op_summary(record)
            summary_stats = summary_drs.record_op_stats[0]
            summary = summary_drs.data_records[0].summary

            summary_record = DataRecord(
                data_item=Summary(summary=summary),
                source_indices=filename,
            )
            f1_drs = op_f1(record)
            f1_stats = f1_drs.record_op_stats[0]
            is_f1 = f1_drs.data_records[0].is_f1

            print(f"has_f1={is_f1}", flush=True)
            latency = [summary_stats.llm_call_duration_secs, f1_stats.llm_call_duration_secs]
            op_latency = [op_latency[i] + latency[i] for i in range(2)]
            total_latency += sum(latency)
            
            cost = [summary_stats.total_input_cost + summary_stats.total_output_cost,
                    f1_stats.total_input_cost + f1_stats.total_output_cost]
            op_cost = [op_cost[i] + cost[i] for i in range(2)]
            total_cost += sum(cost)
            if is_f1 == gt_is_f1:
                correct += 1

        quality = correct / len(paper_dataset)
        print(
            f"quality={quality:.4f}  llm_latency={total_latency:.2f}s  "
            f"cost=${total_cost:.6f}  correct={correct}/{len(paper_dataset)}",
            flush=True,
        )
        return {
            "plan_label": plan_label,
            "quality": quality,
            "total_latency_secs": total_latency,
            "total_cost_usd": total_cost,
            "op_latency_secs": op_latency,
            "op_cost_usd": op_cost,
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
            "op_latency_secs": None,
            "op_cost_usd": None,
            "correct": None,
            "n_papers": len(paper_dataset),
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Main loop: 17 summarizer configs × 17 classifier configs = 289 plans
# ---------------------------------------------------------------------------

# all_results = []
# N_TOTAL = len(OP_CONFIGS) ** 2  # 289

# for global_idx, (sum_cfg, cls_cfg) in enumerate(iproduct(OP_CONFIGS, OP_CONFIGS), start=1):
#     label = f"summarizer={sum_cfg['label']} | classifier={cls_cfg['label']}"

#     op_summarizer = make_op(sum_cfg, TextFile,  PaperFile,           "paper_summarizer")
#     op_classifier = make_op(cls_cfg, PaperFile, PaperClassification, "paper_classifier", depends_on=["summary"])

#     result = run_plan(op_summarizer, op_classifier, label, global_idx, N_TOTAL)
#     result["summarizer_impl"] = sum_cfg["label"]
#     result["classifier_impl"] = cls_cfg["label"]
#     all_results.append(result)

#     # Intermediate save every 10 completed plans
#     if global_idx % 10 == 0:
#         out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"paper_results_{global_idx}.csv")
#         pd.DataFrame(all_results).to_csv(out_path, index=False)

# pd.DataFrame(all_results).to_csv("paper_results.csv", index=False)


# ---------------------------------------------------------------------------
# Get summaries from 9 LCB summarizer configs for all 3 papers
# ---------------------------------------------------------------------------
# rows = []
# for i, summary_model in enumerate(all_models):
#     config = f"LCB({summary_model.name})"
#     op_summarizer = LLMConvertBonded(model=summary_model, input_schema=TextFile,
#                 logical_op_id="paper_summarizer", output_schema=Summary)
#     print(f"\n Config {config}:")
#     for filename, record in paper_dataset:
#         drs_summary = op_summarizer(record)
#         summary_stats = drs_summary.record_op_stats[0]
#         summary = drs_summary.data_records[0].summary or ""

#         row = {
#             "summary_config": config,
#             "record": filename,
#             "summary": summary,
#         }
#         rows.append(row)
#         print(f"{filename}: {summary}")

# pd = pd.DataFrame(rows)
# pd.to_csv("paper_summaries.csv", index=False)

# ---------------------------------------------------------------------------
# main loop: get plan results
# ---------------------------------------------------------------------------
combo_idx = 1
num_combos = 9**2
combos = tuple(product(range(9), repeat=2))
plan_strengths = tuple(product(all_models, repeat=2))
results = []

for model_summary, model_f1 in plan_strengths:
    print(combos[combo_idx-1])
    plan_label = (model_summary.name, model_f1.name)
    op_summary = make_op({"type": "LCB", "model": model_summary},  TextFile, Summary, "summary_extracter")
    op_f1 = make_op({"type": "LCB", "model": model_f1},  Summary, is_F1, "f1_classifier")
    plan_results = run_paper_plan(op_summary, op_f1,
                                  plan_label, combo_idx, num_combos)
    results.append(plan_results)
    combo_idx += 1
df = pd.DataFrame(results)
df.to_csv(f"new_paper_results.csv", index=False)