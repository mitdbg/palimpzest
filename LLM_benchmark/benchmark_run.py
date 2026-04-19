import io
import json
import os
import re
import random
import string
import subprocess
import tarfile
import tempfile
import time
from collections import defaultdict, Counter
from math import comb

import litellm
import pandas as pd
import requests
from bert_score import score as bert_score_fn
from datasets import load_dataset
from rouge_score import rouge_scorer
from src.palimpzest.constants import Model

# ── Models to evaluate ────────────────────────────────────────────────────────
MODELS = [
    Model.GPT_4o,
    # Model.GPT_4o_MINI,
    # Model.GPT_4_1,
    # Model.GPT_4_1_MINI,
    # Model.GPT_4_1_NANO,
    # Model.GPT_5,
    # Model.GPT_5_MINI,
    # Model.GPT_5_NANO,
    # Model.o4_MINI,
]

# ── Benchmark config ───────────────────────────────────────────────────────────
TOTAL_QUESTIONS = 1
RANDOM_SEED = 42

# ── Benchmark registry ─────────────────────────────────────────────────────────
# Each entry maps a benchmark name to its dataset loading args, functions, and output path.
BENCHMARKS = {
    # "mmlu_pro": {
    #     "dataset_name": "TIGER-Lab/MMLU-Pro",
    #     "dataset_kwargs": {"split": "test"},
    #     "sample_fn": lambda ds, n, seed: MMLUPro_sample_questions(ds, n, seed),
    #     "run_fn": lambda models, qs: run_MMLUPro(models, qs),
    #     "output_path": "LLM_benchmark/mmlupro.csv",
    # },
    # "hotpotqa": {
    #     "dataset_name": "hotpot_qa",
    #     "dataset_kwargs": {"name": "distractor", "split": "train", "trust_remote_code": True},
    #     "sample_fn": lambda ds, n, seed: HotpotQA_sample_questions(ds, n, seed),
    #     "run_fn": lambda models, qs: run_HotpotQA(models, qs),
    #     "output_path": "LLM_benchmark/hotpotqa.csv",
    # },
    # "narrativeqa": {
    #     "dataset_name": "deepmind/narrativeqa",
    #     "dataset_kwargs": {"split": "test"},
    #     "sample_fn": lambda ds, n, seed: NarrativeQA_sample_questions(ds, n, seed),
    #     "run_fn": lambda models, qs: run_NarrativeQA(models, qs),
    #     "output_path": "LLM_benchmark/narrativeqa.csv",
    # },
    # "qasper": {
    #     # QASPER uses a custom loader (S3 download) instead of load_dataset
    #     "loader_fn": lambda: QASPER_load_dataset(),
    #     "sample_fn": lambda ds, n, seed: QASPER_sample_questions(ds, n, seed),
    #     "run_fn": lambda models, qs: run_QASPER(models, qs),
    #     "output_path": "LLM_benchmark/qasper.csv",
    # },
    # "fever": {
    #     # FEVER uses a custom loader: local JSONL + HuggingFace wiki pages
    #     "loader_fn": lambda: FEVER_load_dataset(),
    #     "sample_fn": lambda ds, n, seed: FEVER_sample_questions(ds, n, seed),
    #     "run_fn": lambda models, qs: run_FEVER(models, qs),
    #     "output_path": "LLM_benchmark/fever.csv",
    # },
    # "cnn_dailymail": {
    #     "dataset_name": "abisee/cnn_dailymail",
    #     "dataset_kwargs": {"name": "3.0.0", "split": "test"},
    #     "sample_fn": lambda ds, n, seed: CNNDailyMail_sample_questions(ds, n, seed),
    #     "run_fn": lambda models, qs: run_CNNDailyMail(models, qs),
    #     "output_path": "LLM_benchmark/cnn_dailymail.csv",
    # },
    # "math": {
    #     "dataset_name": "nlile/hendrycks-MATH-benchmark",
    #     "dataset_kwargs": {"split": "test"},
    #     "sample_fn": lambda ds, n, seed: MATH_sample_questions(ds, n, seed),
    #     "run_fn": lambda models, qs: run_MATH(models, qs),
    #     "output_path": "LLM_benchmark/math.csv",
    # },
    # "drop": {
    #     "dataset_name": "ucinlp/drop",
    #     "dataset_kwargs": {"split": "validation"},
    #     "sample_fn": lambda ds, n, seed: DROP_sample_questions(ds, n, seed),
    #     "run_fn": lambda models, qs: run_DROP(models, qs),
    #     "output_path": "LLM_benchmark/drop.csv",
    # },
    "humaneval": {
        "dataset_name": "openai/openai_humaneval",
        "dataset_kwargs": {"split": "test"},
        "sample_fn": lambda ds, n, seed: HumanEval_sample_questions(ds, n, seed),
        "run_fn": lambda models, qs: run_HumanEval(models, qs),
        "output_path": "LLM_benchmark/humaneval.csv",
    },
}

# ── Shared helper ──────────────────────────────────────────────────────────────

def _call_model(model: Model, system_prompt: str, user_prompt: str) -> str | None:
    """Call litellm with the right kwargs for the given model. Returns raw response text."""
    completion_kwargs = {}
    if not model.is_o_model() and not model.is_gpt_5_model():
        completion_kwargs["temperature"] = 0.0
    try:
        response = litellm.completion(
            model=model.value,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **completion_kwargs,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    [error] {model}: {e}")
        return None


def _sample_evenly(dataset, group_key_fn, n_total: int, seed: int) -> list[dict]:
    """Generic stratified sampler. group_key_fn maps an item to its stratum key."""
    rng = random.Random(seed)
    by_group: dict = defaultdict(list)
    for item in dataset:
        by_group[group_key_fn(item)].append(item)

    groups = sorted(by_group.keys())
    base, remainder = divmod(n_total, len(groups))
    sampled = []
    for i, group in enumerate(groups):
        n = base + (1 if i < remainder else 0)
        pool = by_group[group]
        sampled.extend(rng.sample(pool, min(n, len(pool))))

    rng.shuffle(sampled)
    return sampled


### ── MMLU-Pro ────────────────────────────────────────────────────────────────

MMLUPro_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant answering multiple-choice questions. "
    "Respond with ONLY the letter of the correct answer (A, B, C, D, E, F, G, H, I, or J). "
    "Do not include any explanation or additional text."
)


def MMLUPro_build_question_prompt(question: str, options: list[str]) -> str:
    option_letters = "ABCDEFGHIJ"
    options_text = "\n".join(f"{option_letters[i]}. {opt}" for i, opt in enumerate(options))
    return f"Question: {question}\n\nOptions:\n{options_text}\n\nAnswer:"


def MMLUPro_sample_questions(dataset, n_total: int, seed: int) -> list[dict]:
    """Sample n_total questions evenly across categories and save to CSV."""
    sampled = _sample_evenly(dataset, lambda item: item["category"], n_total, seed)

    pd.DataFrame([
        {
            "question_id": item.get("question_id", idx),
            "category": item["category"],
            "question": item["question"],
            "correct_answer": item["answer"],
        }
        for idx, item in enumerate(sampled)
    ]).to_csv("LLM_benchmark/mmlupro_questions.csv", index=False)
    print("  Sampled questions saved to LLM_benchmark/mmlupro_questions.csv")

    return sampled


def MMLUPro_query_model(model: Model, item: dict) -> str | None:
    """Return predicted answer letter, or None on error."""
    raw = _call_model(model, MMLUPro_SYSTEM_PROMPT, MMLUPro_build_question_prompt(item["question"], item["options"]))
    if raw is None:
        return None
    for ch in raw:
        if ch in "ABCDEFGHIJ":
            return ch
    return raw


def run_MMLUPro(models: list[Model], questions: list[dict]) -> pd.DataFrame:
    records = []
    for model in models:
        print(f"\nEvaluating {model} on {len(questions)} MMLU-Pro questions...")
        total_correct = 0
        for idx, item in enumerate(questions):
            pred = MMLUPro_query_model(model, item)
            correct = pred == item["answer"]
            if correct:
                total_correct += 1

            records.append({
                "model": model.name,
                "question_id": item.get("question_id", idx),
                "category": item["category"],
                "question": item["question"],
                "correct_answer": item["answer"],
                "predicted_answer": pred,
                "quality": correct,
            })

            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(questions)}] running accuracy: {total_correct/(idx+1):.1%}")

            time.sleep(0.1)

        print(f"  Final accuracy for {model}: {total_correct/len(questions):.1%} ({total_correct}/{len(questions)})")

    return pd.DataFrame(records)


### ── HotpotQA ────────────────────────────────────────────────────────────────

HotpotQA_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant that answers multi-hop questions. "
    "Given several passages and a question, respond with a JSON object with exactly two keys:\n"
    '  "answer": a short answer string (a word or phrase)\n'
    '  "supporting_facts": a list of [passage_title, sentence_index] pairs that are the '
    "key sentences you used to arrive at the answer.\n"
    "Output only the JSON object, no additional text."
)

def HotpotQA_build_question_prompt(question: str, context: dict) -> str:
    passages = []
    for title, sentences in zip(context["title"], context["sentences"]):
        numbered = " ".join(f"[{i}] {s}" for i, s in enumerate(sentences))
        passages.append(f"Title: {title}\n{numbered}")
    return "\n\n".join(passages) + f"\n\nQuestion: {question}\n\nAnswer (JSON only):"

def _normalize_answer(text) -> str:
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = sum(pred_tokens.count(t) for t in common) / len(pred_tokens)
    recall = sum(gold_tokens.count(t) for t in common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _sp_metrics(pred_sp: list, gold_sp: dict) -> tuple[float, float]:
    gold_set = set(zip(gold_sp["title"], gold_sp["sent_id"]))
    pred_set = set(tuple(p) for p in pred_sp) if pred_sp else set()
    if not gold_set and not pred_set:
        return 1.0, 1.0
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0
    sp_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    sp_em = 1.0 if pred_set == gold_set else 0.0
    return sp_em, sp_f1


def HotpotQA_sample_questions(dataset, n_total: int, seed: int) -> list[dict]:
    """Sample n_total questions evenly across (type, level) combos and save to CSV."""
    sampled = _sample_evenly(dataset, lambda item: (item["type"], item["level"]), n_total, seed)

    pd.DataFrame([
        {
            "id": item["id"],
            "type": item["type"],
            "level": item["level"],
            "question": item["question"],
            "answer": item["answer"],
            "supporting_facts": json.dumps(list(zip(item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"]))),
        }
        for item in sampled
    ]).to_csv("LLM_benchmark/hotpotqa_questions.csv", index=False)
    print("Sampled questions saved to LLM_benchmark/hotpotqa_questions.csv")

    return sampled


def HotpotQA_query_model(model: Model, item: dict) -> tuple[str, list]:
    """Return (predicted_answer, predicted_supporting_facts)."""
    raw = _call_model(model, HotpotQA_SYSTEM_PROMPT, HotpotQA_build_question_prompt(item["question"], item["context"]))
    if raw is None:
        return "", []
    try:
        parsed = json.loads(raw)
        return parsed.get("answer", ""), parsed.get("supporting_facts", [])
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                return parsed.get("answer", ""), parsed.get("supporting_facts", [])
            except Exception:
                pass
        print(f"    [parse error] {model}: could not parse JSON from: {raw[:100]}")
        return "", []


def run_HotpotQA(models: list[Model], questions: list[dict]) -> pd.DataFrame:
    records = []
    for model in models:
        print(f"\nEvaluating {model} on {len(questions)} HotpotQA questions...")
        sums = {"ans_em": 0.0, "ans_f1": 0.0, "sp_em": 0.0, "sp_f1": 0.0, "joint_em": 0.0, "joint_f1": 0.0}

        for idx, item in enumerate(questions):
            pred_answer, pred_sp = HotpotQA_query_model(model, item)
            gold_answer = item["answer"]

            ans_em = 1.0 if _normalize_answer(pred_answer) == _normalize_answer(gold_answer) else 0.0
            ans_f1 = _token_f1(pred_answer, gold_answer)
            sp_em, sp_f1 = _sp_metrics(pred_sp, item["supporting_facts"])
            joint_em = ans_em * sp_em
            joint_f1 = ans_f1 * sp_f1

            for k, v in zip(sums, [ans_em, ans_f1, sp_em, sp_f1, joint_em, joint_f1]):
                sums[k] += v

            records.append({
                "model": model.name,
                "ans_em": ans_em,
                "ans_f1": ans_f1,
                "sp_em": sp_em,
                "sp_f1": sp_f1,
                "joint_em": joint_em,
                "joint_f1": joint_f1,
                "id": item["id"],
                "type": item["type"],
                "level": item["level"],
                "question": item["question"],
                "gold_answer": gold_answer,
                "predicted_answer": pred_answer,
                "gold_sp": json.dumps(list(zip(item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"]))),
                "predicted_sp": json.dumps(pred_sp),
            })

            if (idx + 1) % 10 == 0:
                n = idx + 1
                print(
                    f"  [{n}/{len(questions)}] "
                    f"ans_EM={sums['ans_em']/n:.1%}  ans_F1={sums['ans_f1']/n:.1%}  "
                    f"joint_EM={sums['joint_em']/n:.1%}  joint_F1={sums['joint_f1']/n:.1%}"
                )

            time.sleep(0.1)

        n = len(questions)
        print(
            f"  Final [{model}] "
            f"ans_EM={sums['ans_em']/n:.1%}  ans_F1={sums['ans_f1']/n:.1%}  "
            f"sp_EM={sums['sp_em']/n:.1%}  sp_F1={sums['sp_f1']/n:.1%}  "
            f"joint_EM={sums['joint_em']/n:.1%}  joint_F1={sums['joint_f1']/n:.1%}"
        )

    return pd.DataFrame(records)


### ── NarrativeQA ─────────────────────────────────────────────────────────────

NarrativeQA_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant that answers questions about stories and movies based on a provided summary. "
    "Answer with one word, a few-word phrase, or a short sentence. "
    "Avoid extra, unnecessary information in the answer. "
    "Do not include 'Answer:' in your answer."
)


def NarrativeQA_build_question_prompt(question: str, summary: str) -> str:
    return f"Summary:\n{summary}\n\nQuestion: {question}"


def NarrativeQA_sample_questions(dataset, n_total: int, seed: int) -> list[dict]:
    """Sample n_total questions evenly across document kinds (movie, gutenberg) and save to CSV."""
    sampled = _sample_evenly(dataset, lambda item: item["document"]["kind"], n_total, seed)

    pd.DataFrame([
        {
            "id": item["document"]["id"],
            "kind": item["document"]["kind"],
            "question": item["question"]["text"],
            "answer_1": item["answers"][0]["text"],
            "answer_2": item["answers"][1]["text"],
        }
        for item in sampled
    ]).to_csv("LLM_benchmark/narrativeqa_questions.csv", index=False)
    print("  Sampled questions saved to LLM_benchmark/narrativeqa_questions.csv")

    return sampled


def NarrativeQA_query_model(model: Model, item: dict) -> str | None:
    """Return the predicted answer string."""
    summary = item["document"]["summary"]["text"]
    question = item["question"]["text"]
    return _call_model(model, NarrativeQA_SYSTEM_PROMPT, NarrativeQA_build_question_prompt(question, summary))


def run_NarrativeQA(models: list[Model], questions: list[dict]) -> pd.DataFrame:
    records = []
    for model in models:
        print(f"\nEvaluating {model} on {len(questions)} NarrativeQA questions...")
        f1_sum = 0.0

        for idx, item in enumerate(questions):
            gold_answers = [ans["text"] for ans in item["answers"]]  # always 2 references
            pred = NarrativeQA_query_model(model, item)
            if pred is None:
                pred = ""

            # take the max F1 over the two reference answers
            max_f1 = max(_token_f1(pred, gold) for gold in gold_answers)
            f1_sum += max_f1

            records.append({
                "model": model.name,
                "max_f1": max_f1,
                "id": item["document"]["id"],
                "kind": item["document"]["kind"],
                "question": item["question"]["text"],
                "gold_answer_1": gold_answers[0],
                "gold_answer_2": gold_answers[1],
                "predicted_answer": pred,
            })

            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(questions)}] running avg F1: {f1_sum/(idx+1):.3f}")

            time.sleep(0.1)

        print(f"  Final avg max-F1 for {model}: {f1_sum/len(questions):.3f}")

    return pd.DataFrame(records)


### ── QASPER ──────────────────────────────────────────────────────────────────

QASPER_S3_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"

QASPER_SYSTEM_PROMPT = (
    "You are a knowledgeable assistant that answers questions about scientific NLP papers based on the provided paper text and figure/table captions. "
    "Answer with one word, a few-word phrase, or a short sentence. "
    "Avoid extra, unnecessary information in the answer. "
    "Do not include 'Answer:' in your answer. "
    "If the question can be answered with yes or no, answer with 'True' for yes or 'False' for no."
    "If the question cannot be answered from the paper, respond with 'Unanswerable'."
)


def QASPER_load_dataset() -> list[dict]:
    """Download QASPER test set from S3 and flatten into one dict per question."""
    print("  Downloading QASPER from S3...")
    r = requests.get(QASPER_S3_URL)
    tf = tarfile.open(fileobj=io.BytesIO(r.content))
    data = json.loads(tf.extractfile("qasper-test-v0.3.json").read())

    def _dominant_answer_type(answers: list[dict]) -> str:
        types = []
        for ann in answers:
            a = ann["answer"]
            if a["unanswerable"]:
                types.append("unanswerable")
            elif a["yes_no"] is not None:
                types.append("yes_no")
            elif a["free_form_answer"]:
                types.append("free_form")
            else:
                types.append("extractive")
        return Counter(types).most_common(1)[0][0] if types else "extractive"

    def _extract_gold_texts(answers: list[dict]) -> list[str]:
        """Return one gold answer string per annotator (skip unanswerable annotations)."""
        texts = []
        for ann in answers:
            a = ann["answer"]
            if a["unanswerable"]:
                texts.append("unanswerable")
            elif a["yes_no"] is not None:
                texts.append(str(a["yes_no"]))
            elif a["free_form_answer"]:
                texts.append(a["free_form_answer"])
            elif a["extractive_spans"]:
                texts.append(" ".join(a["extractive_spans"]))
        return texts

    def _format_full_text(full_text: list[dict]) -> str:
        sections = []
        for section in full_text:
            header = section["section_name"] or "Body"
            body = "\n".join(section["paragraphs"])
            sections.append(f"## {header}\n{body}")
        return "\n\n".join(sections)

    def _format_figures(figures_and_tables: list[dict]) -> str:
        captions = [f"[{fig['file']}] {fig['caption']}" for fig in figures_and_tables if fig.get("caption")]
        return "\n".join(captions)

    questions = []
    for paper_id, paper in data.items():
        full_text_str = _format_full_text(paper["full_text"])
        figures_str = _format_figures(paper["figures_and_tables"])
        for qa in paper["qas"]:
            questions.append({
                "paper_id": paper_id,
                "title": paper["title"],
                "full_text": full_text_str,
                "figures_and_tables": figures_str,
                "question_id": qa["question_id"],
                "question": qa["question"],
                "dominant_type": _dominant_answer_type(qa["answers"]),
                "gold_answers": _extract_gold_texts(qa["answers"]),
            })
    return questions


def QASPER_build_question_prompt(question: str, full_text: str, figures_and_tables: str) -> str:
    parts = [f"Paper:\n{full_text}"]
    if figures_and_tables:
        parts.append(f"Figures and Tables:\n{figures_and_tables}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def QASPER_sample_questions(questions: list[dict], n_total: int, seed: int) -> list[dict]:
    """Sample n_total questions evenly across dominant answer types and save to CSV."""
    sampled = _sample_evenly(questions, lambda item: item["dominant_type"], n_total, seed)

    pd.DataFrame([
        {
            "paper_id": item["paper_id"],
            "question_id": item["question_id"],
            "title": item["title"],
            "dominant_type": item["dominant_type"],
            "question": item["question"],
            "gold_answers": json.dumps(item["gold_answers"]),
        }
        for item in sampled
    ]).to_csv("LLM_benchmark/qasper_questions.csv", index=False)
    print("  Sampled questions saved to LLM_benchmark/qasper_questions.csv")

    return sampled


def QASPER_query_model(model: Model, item: dict) -> str | None:
    return _call_model(model, QASPER_SYSTEM_PROMPT, QASPER_build_question_prompt(item["question"], item["full_text"], item["figures_and_tables"]))


def _qasper_normalize(text: str) -> str:
    """Normalize yes/no to true/false before F1 computation."""
    normalized = _normalize_answer(text)
    if normalized == "yes":
        return "true"
    if normalized == "no":
        return "false"
    return normalized


def _qasper_token_f1(pred: str, gold: str) -> float:
    pred_tokens = _qasper_normalize(pred).split()
    gold_tokens = _qasper_normalize(gold).split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0
    precision = sum(pred_tokens.count(t) for t in common) / len(pred_tokens)
    recall = sum(gold_tokens.count(t) for t in common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def run_QASPER(models: list[Model], questions: list[dict]) -> pd.DataFrame:
    records = []
    for model in models:
        print(f"\nEvaluating {model} on {len(questions)} QASPER questions...")
        f1_sum = 0.0

        for idx, item in enumerate(questions):
            pred = QASPER_query_model(model, item)
            if pred is None:
                pred = ""

            max_f1 = max(_qasper_token_f1(pred, gold) for gold in item["gold_answers"]) if item["gold_answers"] else 0.0
            f1_sum += max_f1

            records.append({
                "model": model.name,
                "max_f1": max_f1,
                "paper_id": item["paper_id"],
                "question_id": item["question_id"],
                "title": item["title"],
                "dominant_type": item["dominant_type"],
                "question": item["question"],
                "gold_answers": json.dumps(item["gold_answers"]),
                "predicted_answer": pred,
            })

            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(questions)}] running avg max-F1: {f1_sum/(idx+1):.3f}")

            time.sleep(0.1)

        print(f"  Final avg max-F1 for {model}: {f1_sum/len(questions):.3f}")

    return pd.DataFrame(records)


### ── FEVER ───────────────────────────────────────────────────────────────────

FEVER_SYSTEM_PROMPT = (
    "You are a fact-checking assistant. "
    "You will be given a claim and relevant Wikipedia sentences as evidence. "
    "Determine whether the evidence SUPPORTS or REFUTES the claim, "
    "or if there is NOT ENOUGH INFO to make a determination.\n"
    "Respond with a JSON object with exactly two keys:\n"
    '  "label": one of "SUPPORTS", "REFUTES", or "NOT ENOUGH INFO"\n'
    '  "evidence": a list of [page_title, sentence_index] pairs identifying the key sentences '
    "that support your verdict (empty list for NOT ENOUGH INFO)\n"
    "Output only the JSON object, no additional text."
)


def FEVER_load_dataset() -> list[dict]:
    """Load FEVER claims from local JSONL and attach oracle Wikipedia sentences to each."""
    print("  Loading FEVER claims from 'train (1).jsonl'...")
    claims = []
    with open("train (1).jsonl") as f:
        for line in f:
            claims.append(json.loads(line))
    print(f"  Loaded {len(claims)} claims.")

    # Collect all unique Wikipedia page titles referenced across all evidence
    needed_titles: set[str] = set()
    for item in claims:
        for ev_group in item["evidence"]:
            for ev in ev_group:
                if ev[2] is not None:
                    needed_titles.add(ev[2])
    print(f"  Need wiki pages for {len(needed_titles)} unique titles. Loading from HuggingFace (this may take a few minutes)...")

    wiki_ds = load_dataset("fever/fever", "wiki_pages", split="wikipedia_pages", trust_remote_code=True)
    wiki_ds = wiki_ds.filter(lambda batch: [id_ in needed_titles for id_ in batch["id"]], batched=True)

    # Build lookup: page_title -> {sent_id: sent_text}
    wiki_lookup: dict[str, dict[int, str]] = {}
    for page in wiki_ds:
        sents: dict[int, str] = {}
        for line in page["lines"].split("\n"):
            parts = line.split("\t")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1]:
                sents[int(parts[0])] = parts[1]
        wiki_lookup[page["id"]] = sents
    print(f"  Built wiki lookup for {len(wiki_lookup)} pages.")

    # Attach oracle sentences and gold evidence sets to each claim
    questions = []
    for item in claims:
        # Gold evidence: one frozenset of (title, sent_id) per annotator group
        gold_ev_sets: list[frozenset] = []
        for ev_group in item["evidence"]:
            ev_set = frozenset(
                (ev[2], ev[3])
                for ev in ev_group
                if ev[2] is not None and ev[3] is not None
            )
            if ev_set:
                gold_ev_sets.append(ev_set)

        # Collect all referenced page titles (union across all annotator groups)
        page_titles: set[str] = set()
        for ev_group in item["evidence"]:
            for ev in ev_group:
                if ev[2] is not None:
                    page_titles.add(ev[2])

        # Format ALL sentences from each referenced page
        by_page: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for title in page_titles:
            for sent_id, text in wiki_lookup.get(title, {}).items():
                by_page[title].append((sent_id, text))

        wiki_text_parts = []
        for title in sorted(by_page):
            numbered = "\n".join(f"[{sid}] {txt}" for sid, txt in sorted(by_page[title]))
            wiki_text_parts.append(f"Title: {title}\n{numbered}")
        wiki_text = "\n\n".join(wiki_text_parts)

        questions.append({
            "id": item["id"],
            "verifiable": item["verifiable"],
            "label": item["label"],
            "claim": item["claim"],
            "gold_ev_sets": gold_ev_sets,
            "wiki_text": wiki_text,
        })

    return questions


def FEVER_sample_questions(questions: list[dict], n_total: int, seed: int) -> list[dict]:
    """Sample n_total questions evenly across label types and save to CSV."""
    sampled = _sample_evenly(questions, lambda item: item["label"], n_total, seed)

    pd.DataFrame([
        {
            "id": item["id"],
            "verifiable": item["verifiable"],
            "label": item["label"],
            "claim": item["claim"],
            "gold_evidence": json.dumps([list(s) for s in item["gold_ev_sets"]]),
        }
        for item in sampled
    ]).to_csv("LLM_benchmark/fever_questions.csv", index=False)
    print("  Sampled questions saved to LLM_benchmark/fever_questions.csv")

    return sampled


def FEVER_build_question_prompt(claim: str, wiki_text: str) -> str:
    parts = []
    if wiki_text:
        parts.append(f"Wikipedia Evidence:\n{wiki_text}")
    parts.append(f"Claim: {claim}\n\nAnswer (JSON only):")
    return "\n\n".join(parts)


def FEVER_query_model(model: Model, item: dict) -> tuple[str, list]:
    """Return (predicted_label, predicted_evidence) where evidence is [[title, sent_id], ...]."""
    raw = _call_model(model, FEVER_SYSTEM_PROMPT, FEVER_build_question_prompt(item["claim"], item["wiki_text"]))
    if raw is None:
        return "", []
    try:
        parsed = json.loads(raw)
        return parsed.get("label", "").upper(), parsed.get("evidence", [])
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                return parsed.get("label", "").upper(), parsed.get("evidence", [])
            except Exception:
                pass
        print(f"    [parse error] {model}: could not parse JSON from: {raw[:100]}")
        return "", []


def _fever_evidence_metrics(pred_ev: list, gold_ev_sets: list[frozenset]) -> tuple[float, float, float]:
    """Return (precision, recall, F1) against the best-matching gold evidence set."""
    pred_set = frozenset(tuple(e) for e in pred_ev)
    if not gold_ev_sets:
        return (1.0, 1.0, 1.0) if not pred_set else (0.0, 1.0, 0.0)

    best_p, best_r, best_f1 = 0.0, 0.0, 0.0
    for gold_set in gold_ev_sets:
        tp = len(pred_set & gold_set)
        p = tp / len(pred_set) if pred_set else 0.0
        r = tp / len(gold_set) if gold_set else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        if f1 > best_f1:
            best_p, best_r, best_f1 = p, r, f1

    return best_p, best_r, best_f1


def run_FEVER(models: list[Model], questions: list[dict]) -> pd.DataFrame:
    records = []
    for model in models:
        print(f"\nEvaluating {model} on {len(questions)} FEVER claims...")
        sums = {"label_acc": 0.0, "ev_p": 0.0, "ev_r": 0.0, "ev_f1": 0.0, "fever_score": 0.0}

        for idx, item in enumerate(questions):
            pred_label, pred_ev = FEVER_query_model(model, item)
            gold_label = item["label"]
            label_correct = pred_label == gold_label

            # Evidence metrics only meaningful for VERIFIABLE claims
            if item["verifiable"] == "NOT VERIFIABLE":
                ev_p, ev_r, ev_f1 = float("nan"), float("nan"), float("nan")
                fever_score = float(label_correct)
            else:
                ev_p, ev_r, ev_f1 = _fever_evidence_metrics(pred_ev, item["gold_ev_sets"])
                pred_set = frozenset(tuple(e) for e in pred_ev)
                ev_correct = any(pred_set <= gold_set for gold_set in item["gold_ev_sets"]) if item["gold_ev_sets"] else True
                fever_score = float(label_correct and ev_correct)
                sums["ev_p"] += ev_p
                sums["ev_r"] += ev_r
                sums["ev_f1"] += ev_f1

            sums["label_acc"] += float(label_correct)
            sums["fever_score"] += fever_score

            records.append({
                "model": model.name,
                "id": item["id"],
                "verifiable": item["verifiable"],
                "gold_label": gold_label,
                "predicted_label": pred_label,
                "label_correct": label_correct,
                "ev_precision": ev_p,
                "ev_recall": ev_r,
                "ev_f1": ev_f1,
                "fever_score": fever_score,
                "claim": item["claim"],
                "gold_evidence": json.dumps([list(s) for s in item["gold_ev_sets"]]),
                "predicted_evidence": json.dumps(pred_ev),
            })

            if (idx + 1) % 10 == 0:
                n = idx + 1
                n_ver = sum(1 for r in records[-n:] if r["verifiable"] == "VERIFIABLE")
                print(
                    f"  [{n}/{len(questions)}] "
                    f"label_acc={sums['label_acc']/n:.1%}  "
                    f"ev_F1={sums['ev_f1']/n_ver:.1%}  "
                    f"FEVER={sums['fever_score']/n:.1%}"
                )

            time.sleep(0.1)

        n = len(questions)
        n_ver = sum(1 for r in records if r["model"] == model.name and r["verifiable"] == "VERIFIABLE")
        print(
            f"  Final [{model}] "
            f"label_acc={sums['label_acc']/n:.1%}  "
            f"ev_P={sums['ev_p']/n_ver:.1%}  ev_R={sums['ev_r']/n_ver:.1%}  ev_F1={sums['ev_f1']/n_ver:.1%}  "
            f"FEVER_score={sums['fever_score']/n:.1%}"
        )

    return pd.DataFrame(records)


### ── CNN/DailyMail ───────────────────────────────────────────────────────────

CNNDailyMail_SYSTEM_PROMPT = (
    "You are a professional news summarizer. "
    "Given a news article, write a concise multi-sentence summary that captures the key facts and main points. "
    "Output only the summary text, with no preamble, labels, or extra commentary."
)

_rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def CNNDailyMail_build_prompt(article: str) -> str:
    return f"Article:\n{article}\n\nSummary:"


def CNNDailyMail_sample_questions(dataset, n_total: int, seed: int) -> list[dict]:
    """Randomly sample n_total articles and save to CSV."""
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), n_total)
    sampled = [dataset[i] for i in indices]

    pd.DataFrame([
        {
            "id": item["id"],
            "article": item["article"],
            "highlights": item["highlights"],
        }
        for item in sampled
    ]).to_csv("LLM_benchmark/cnn_dailymail_questions.csv", index=False)
    print("  Sampled questions saved to LLM_benchmark/cnn_dailymail_questions.csv")

    return sampled


def CNNDailyMail_query_model(model: Model, item: dict) -> str | None:
    return _call_model(model, CNNDailyMail_SYSTEM_PROMPT, CNNDailyMail_build_prompt(item["article"]))


def run_CNNDailyMail(models: list[Model], questions: list[dict]) -> pd.DataFrame:
    records = []
    for model in models:
        print(f"\nEvaluating {model} on {len(questions)} CNN/DailyMail articles...")

        predictions, references, ids = [], [], []
        per_item: list[dict] = []

        for idx, item in enumerate(questions):
            pred = CNNDailyMail_query_model(model, item)
            if pred is None:
                pred = ""

            gold = item["highlights"]
            scores = _rouge_scorer.score(gold, pred)

            per_item.append({
                "model": model.name,
                "id": item["id"],
                "rouge1_f1": scores["rouge1"].fmeasure,
                "rouge2_f1": scores["rouge2"].fmeasure,
                "rougeL_f1": scores["rougeL"].fmeasure,
                "article": item["article"],
                "highlights": gold,
                "predicted_summary": pred,
            })
            predictions.append(pred)
            references.append(gold)
            ids.append(item["id"])

            if (idx + 1) % 10 == 0:
                n = idx + 1
                avg_r1 = sum(r["rouge1_f1"] for r in per_item) / n
                avg_r2 = sum(r["rouge2_f1"] for r in per_item) / n
                avg_rl = sum(r["rougeL_f1"] for r in per_item) / n
                print(
                    f"  [{n}/{len(questions)}] "
                    f"ROUGE-1={avg_r1:.3f}  ROUGE-2={avg_r2:.3f}  ROUGE-L={avg_rl:.3f}"
                )

            time.sleep(0.1)

        # BERTScore over all predictions at once (more efficient than per-item)
        print(f"  Computing BERTScore for {model}...")
        P, R, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)
        bert_f1_list = F1.tolist()

        for i, row in enumerate(per_item):
            row["bertscore_f1"] = bert_f1_list[i]
            records.append(row)

        n = len(questions)
        avg_r1 = sum(r["rouge1_f1"] for r in per_item) / n
        avg_r2 = sum(r["rouge2_f1"] for r in per_item) / n
        avg_rl = sum(r["rougeL_f1"] for r in per_item) / n
        avg_bs = sum(bert_f1_list) / n
        print(
            f"  Final [{model}] "
            f"ROUGE-1={avg_r1:.3f}  ROUGE-2={avg_r2:.3f}  ROUGE-L={avg_rl:.3f}  BERTScore-F1={avg_bs:.3f}"
        )

    return pd.DataFrame(records)


### ── Hendrycks MATH ──────────────────────────────────────────────────────────

MATH_SYSTEM_PROMPT = (
    "You are an expert mathematician. Solve the given math problem step by step. "
    "At the end of your solution, provide the final answer in a single boxed expression using LaTeX: \\boxed{...} — for example: "
    "\\boxed{42}, \\boxed{\\frac{3}{4}}, or \\boxed{x^2+1}. "
    "Do not include any text after the \\boxed{} expression."
    "Use exact forms, unless the problem explicitly asks otherwise."
)


def _extract_boxed(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} in text, handling nested braces."""
    # Find the last occurrence so we pick the final answer if the model reasons step-by-step
    idx = text.rfind(r"\boxed{")
    if idx == -1:
        idx = text.rfind(r"\boxed {")
        if idx == -1:
            return None
        start = idx + len(r"\boxed {")
    else:
        start = idx + len(r"\boxed{")

    depth, pos = 1, start
    while pos < len(text) and depth > 0:
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
        pos += 1

    return text[start : pos - 1].strip() if depth == 0 else None


def _normalize_math_str(expr: str) -> str:
    """
    Normalize a LaTeX math string for surface-level comparison.
    Strips surrounding dollars/spaces, removes display-only decorations,
    and collapses all whitespace.
    """
    s = expr.strip().strip("$").strip()
    # Remove \left / \right sizing decorators
    s = re.sub(r"\\left\s*", "", s)
    s = re.sub(r"\\right\s*", "", s)
    # Remove spacing commands
    s = re.sub(r"\\[,;:!]", "", s)
    s = re.sub(r"\\\s", "", s)
    # Normalise \text{...} and \mathrm{...} casing → lower
    s = re.sub(r"\\(?:text|mathrm|mbox)\{([^}]*)\}", lambda m: m.group(1).lower(), s)
    # Collapse all whitespace
    s = re.sub(r"\s+", "", s)
    return s.lower()


def _latex_to_sympy(expr: str):
    """
    Try to parse a LaTeX string into a SymPy expression.
    Applies several lightweight cleanups before giving up.
    Returns None on failure.
    """
    from sympy.parsing.latex import parse_latex

    candidates = [expr]
    # Cleanup variants to attempt
    cleaned = expr
    cleaned = re.sub(r"\\%", "/100", cleaned)
    cleaned = re.sub(r"\\text\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", cleaned)
    cleaned = re.sub(r"\\left|\\right", "", cleaned)
    candidates.append(cleaned)

    for candidate in candidates:
        try:
            return parse_latex(candidate)
        except Exception:
            pass
    return None


def _answers_equivalent(pred: str, gold: str) -> bool:
    """
    Decide whether pred and gold represent the same mathematical answer.

    Tries four strategies in order:
      1. Normalised string equality  (fast, handles most exact matches)
      2. SymPy symbolic difference == 0  (handles algebraic equivalence)
      3. Numerical evaluation with relative tolerance 1e-6
         (handles irrational / decimal forms that differ symbolically)
      4. Plain-float parse  (fallback when LaTeX parsing fails entirely)
    """
    import sympy

    # ── 1. Normalised string match ────────────────────────────────────────────
    if _normalize_math_str(pred) == _normalize_math_str(gold):
        return True

    # ── 2 & 3. SymPy-based checks ─────────────────────────────────────────────
    pred_sym = _latex_to_sympy(pred)
    gold_sym = _latex_to_sympy(gold)

    if pred_sym is not None and gold_sym is not None:
        # 2a. Symbolic difference
        try:
            if sympy.simplify(pred_sym - gold_sym) == 0:
                return True
        except Exception:
            pass

        # 2b. Ratio (catches forms like 2/4 vs 1/2)
        try:
            if sympy.simplify(pred_sym / gold_sym) == 1:
                return True
        except Exception:
            pass

        # 3. Numerical evaluation
        try:
            pred_val = complex(pred_sym.evalf(50))
            gold_val = complex(gold_sym.evalf(50))
            tol = 1e-6 * (1 + abs(gold_val))
            if abs(pred_val - gold_val) < tol:
                return True
        except Exception:
            pass

    # ── 4. Plain float fallback ───────────────────────────────────────────────
    try:
        pred_f = float(re.sub(r"[,\s]", "", pred))
        gold_f = float(re.sub(r"[,\s]", "", gold))
        if abs(pred_f - gold_f) < 1e-6 * (1 + abs(gold_f)):
            return True
    except Exception:
        pass

    return False


def _math_get_answer(item: dict) -> str:
    """
    Return the gold answer string for a MATH dataset item.
    Prefers the pre-extracted 'answer' field; falls back to extracting
    \\boxed{} from 'solution'.
    """
    if "answer" in item and item["answer"]:
        return item["answer"].strip()
    solution = item.get("solution", "")
    extracted = _extract_boxed(solution)
    return extracted if extracted is not None else solution.strip()


def MATH_build_prompt(problem: str) -> str:
    return f"Problem:\n{problem}\n\nSolution:"


def MATH_sample_questions(dataset, n_total: int, seed: int) -> list[dict]:
    """Sample n_total problems evenly across (level, subject) strata.

    If any stratum has fewer items than its quota, the shortfall is filled by
    randomly drawing from the remaining unsampled items so the total is always
    exactly n_total.
    """
    sampled = _sample_evenly(dataset, lambda item: (item["level"], item["subject"]), n_total, seed)

    shortfall = n_total - len(sampled)
    if shortfall > 0:
        rng = random.Random(seed)
        sampled_set = {id(item) for item in sampled}
        remaining = [item for item in dataset if id(item) not in sampled_set]
        sampled.extend(rng.sample(remaining, min(shortfall, len(remaining))))
        rng.shuffle(sampled)

    pd.DataFrame([
        {
            "level": item["level"],
            "subject": item["subject"],
            "problem": item["problem"],
            "gold_answer": _math_get_answer(item),
        }
        for item in sampled
    ]).to_csv("LLM_benchmark/math_questions.csv", index=False)
    print("  Sampled questions saved to LLM_benchmark/math_questions.csv")

    return sampled


def MATH_query_model(model: Model, item: dict) -> tuple[str | None, str | None]:
    """Return (raw_response, extracted_boxed_answer)."""
    raw = _call_model(model, MATH_SYSTEM_PROMPT, MATH_build_prompt(item["problem"]))
    if raw is None:
        return None, None
    return raw, _extract_boxed(raw)


def run_MATH(models: list[Model], questions: list[dict]) -> pd.DataFrame:
    records = []
    for model in models:
        print(f"\nEvaluating {model} on {len(questions)} MATH problems...")
        total_correct = 0

        # Track per-stratum stats for the final breakdown
        by_level: dict[str, list[bool]] = defaultdict(list)
        by_subject: dict[str, list[bool]] = defaultdict(list)

        for idx, item in enumerate(questions):
            gold_answer = _math_get_answer(item)
            raw, pred_answer = MATH_query_model(model, item)

            if pred_answer is None:
                quality = 0
            else:
                quality = int(_answers_equivalent(pred_answer, gold_answer))

            total_correct += quality
            by_level[item["level"]].append(quality)
            by_subject[item["subject"]].append(quality)

            records.append({
                "model": model.name,
                "quality": quality,
                "level": item["level"],
                "subject": item["subject"],
                "problem": item["problem"],
                "gold_answer": gold_answer,
                "predicted_answer": pred_answer if pred_answer is not None else "",
                "raw_response": raw if raw is not None else "",
            })

            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(questions)}] running accuracy: {total_correct/(idx+1):.1%}")

            time.sleep(0.1)

        n = len(questions)
        print(f"  Final accuracy [{model}]: {total_correct/n:.1%} ({total_correct}/{n})")
        print("  By level: " + "  ".join(
            f"{lvl}={sum(v)/len(v):.0%}" for lvl, v in sorted(by_level.items())
        ))
        print("  By subject: " + "  ".join(
            f"{t}={sum(v)/len(v):.0%}" for t, v in sorted(by_subject.items())
        ))

    return pd.DataFrame(records)

### ── DROP ────────────────────────────────────────────────────────────────────

DROP_SYSTEM_PROMPT = (
    "You are a reading comprehension assistant and math expert. "
    "Given a passage and a question, answer with only the answer itself — "
    "a number if possible, otherwise a word or a short phrase. Do not include the unit for numerical answers."
    "Do not include 'Answer:' or any explanation."
)


def _drop_dominant_type(types: list[str]) -> str:
    """Return the most common answer type ('number' or 'spans') across annotators."""
    counts = Counter(types)
    return counts.most_common(1)[0][0] if counts else "spans"


def _drop_gold_texts(answer_dict: dict) -> list[str]:
    """
    Return one gold answer string per annotator.
    Each annotator's answer is the spans joined by a space.
    """
    texts = []
    for spans in answer_dict["spans"]:
        texts.append(" ".join(spans).strip())
    return texts


def _normalize_number(text: str) -> str:
    """Normalize a number string: strip commas, spaces, leading zeros."""
    text = text.strip().replace(",", "").replace(" ", "")
    try:
        # Normalise to a canonical float/int string
        val = float(text)
        if val == int(val):
            return str(int(val))
        return str(val)
    except ValueError:
        return text.lower()


def DROP_build_question_prompt(passage: str, question: str) -> str:
    return f"Passage:\n{passage}\n\nQuestion: {question}\n\nAnswer:"


def DROP_sample_questions(dataset, n_total: int, seed: int) -> list[dict]:
    """Sample n_total questions evenly across dominant answer types and save to CSV."""
    items = []
    for row in dataset:
        types = row["answers_spans"]["types"]
        dom_type = _drop_dominant_type(types)
        if dom_type not in ("spans", "number"):
            continue
        gold_texts = _drop_gold_texts(row["answers_spans"])
        items.append({
            "query_id": row["query_id"],
            "section_id": row["section_id"],
            "passage": row["passage"],
            "question": row["question"],
            "dominant_type": dom_type,
            "gold_texts": gold_texts,
        })

    sampled = _sample_evenly(items, lambda item: item["dominant_type"], n_total, seed)

    pd.DataFrame([
        {
            "query_id": item["query_id"],
            "section_id": item["section_id"],
            "dominant_type": item["dominant_type"],
            "question": item["question"],
            "gold_texts": json.dumps(item["gold_texts"]),
        }
        for item in sampled
    ]).to_csv("LLM_benchmark/drop_questions.csv", index=False)
    print("  Sampled questions saved to LLM_benchmark/drop_questions.csv")

    return sampled


def DROP_query_model(model: Model, item: dict) -> str | None:
    return _call_model(model, DROP_SYSTEM_PROMPT, DROP_build_question_prompt(item["passage"], item["question"]))


def _drop_token_f1(pred: str, gold: str) -> float:
    """Token F1 between normalized pred and gold (used for 'spans' type)."""
    return _token_f1(pred, gold)


def _drop_exact_match(pred: str, gold: str) -> float:
    """Exact match after number normalization (used for 'number' type)."""
    return 1.0 if _normalize_number(pred) == _normalize_number(gold) else 0.0


def run_DROP(models: list[Model], questions: list[dict]) -> pd.DataFrame:
    records = []
    for model in models:
        print(f"\nEvaluating {model} on {len(questions)} DROP questions...")
        score_sum = 0.0
        type_sums: dict[str, float] = defaultdict(float)
        type_counts: dict[str, int] = defaultdict(int)

        for idx, item in enumerate(questions):
            pred = DROP_query_model(model, item)
            if pred is None:
                pred = ""

            dom_type = item["dominant_type"]
            gold_texts = item["gold_texts"]

            if dom_type == "spans":
                quality = max(_drop_token_f1(pred, gold) for gold in gold_texts) if gold_texts else 0.0
            else:  # "number"
                quality = max(_drop_exact_match(pred, gold) for gold in gold_texts) if gold_texts else 0.0

            score_sum += quality
            type_sums[dom_type] += quality
            type_counts[dom_type] += 1

            records.append({
                "model": model.name,
                "quality": quality,
                "query_id": item["query_id"],
                "section_id": item["section_id"],
                "dominant_type": dom_type,
                "question": item["question"],
                "gold_texts": json.dumps(gold_texts),
                "predicted_answer": pred,
            })

            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(questions)}] running avg score: {score_sum/(idx+1):.3f}")

            time.sleep(0.1)

        n = len(questions)
        print(f"  Final avg score [{model}]: {score_sum/n:.3f}")
        for t in sorted(type_sums):
            metric = "token-F1" if t == "spans" else "exact-match"
            print(f"    {t} ({metric}): {type_sums[t]/type_counts[t]:.3f} ({type_counts[t]} questions)")

    return pd.DataFrame(records)



### ── HumanEval ───────────────────────────────────────────────────────────────

# Number of completions per problem used to estimate pass@k.
# The original paper uses 200; reduce for faster/cheaper runs.
HUMANEVAL_NUM_SAMPLES = 40
HUMANEVAL_TEMPERATURE = 0.8   # temperature from the original Chen et al. paper
HUMANEVAL_TIMEOUT_SEC = 10    # per-execution wall-clock timeout

HumanEval_SYSTEM_PROMPT = (
    "You are an expert Python programmer. "
    "Given a Python function stub with a docstring, output the complete function implementation. "
    "Include the def line and the full function body. "
    "Output only valid Python code — no markdown, no explanation."
)


def _estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from Chen et al. 2021: 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def _extract_full_code(prompt: str, response: str, entry_point: str) -> str:
    """Return a self-contained Python source to execute for one HumanEval sample."""
    code = response
    # Strip markdown fences if present
    if "```" in code:
        blocks = re.findall(r"```(?:python)?\n?(.*?)```", code, re.DOTALL)
        if blocks:
            code = blocks[0].strip()

    # If the response already contains the function definition, use it directly.
    if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", code):
        return code

    # Otherwise the model output is just the body — prepend the prompt stub.
    return prompt + code


def _execute_humaneval(code: str, timeout: int = HUMANEVAL_TIMEOUT_SEC) -> bool:
    """Write code to a temp file, run it in a subprocess, return True iff exit-code == 0."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            ["python", fname],
            capture_output=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
    finally:
        os.unlink(fname)


def _call_model_n(
    model: Model,
    system_prompt: str,
    user_prompt: str,
    n: int,
    temperature: float,
) -> list[str]:
    """
    Generate n completions for one prompt.
    Tries a single batched call (n= parameter); falls back to n sequential calls
    for models or providers that do not support batched sampling.
    """
    try:
        response = litellm.completion(
            model=model.value,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            n=n,
            temperature=temperature,
        )
        return [choice.message.content.strip() for choice in response.choices]
    except Exception:
        pass

    # Sequential fallback
    results = []
    for _ in range(n):
        try:
            response = litellm.completion(
                model=model.value,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            results.append(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"    [error] {model}: {e}")
    return results


def HumanEval_sample_questions(dataset, n_total: int, seed: int) -> list[dict]:
    """
    Return up to n_total problems. HumanEval has 164; if n_total >= 164 all are used.
    Saves a CSV with metadata (no passage text to keep file small).
    """
    items = [dict(row) for row in dataset]
    rng = random.Random(seed)
    if len(items) > n_total:
        items = rng.sample(items, n_total)

    pd.DataFrame([
        {
            "task_id": item["task_id"],
            "entry_point": item["entry_point"],
            "prompt": item["prompt"],
        }
        for item in items
    ]).to_csv("LLM_benchmark/humaneval_questions.csv", index=False)
    print("  Sampled questions saved to LLM_benchmark/humaneval_questions.csv")

    return items


def run_HumanEval(models: list[Model], questions: list[dict]) -> pd.DataFrame:
    """
    For each problem generate HUMANEVAL_NUM_SAMPLES completions, execute each against
    the HumanEval test suite, then report pass@1 / pass@10 / pass@100 using the
    unbiased estimator from Chen et al. 2021.
    """
    ks = [1, 10, 20]
    n_samples = HUMANEVAL_NUM_SAMPLES
    records = []

    for model in models:
        print(
            f"\nEvaluating {model} on {len(questions)} HumanEval problems "
            f"({n_samples} samples each, temperature={HUMANEVAL_TEMPERATURE})..."
        )

        for idx, item in enumerate(questions):
            completions = _call_model_n(
                model,
                HumanEval_SYSTEM_PROMPT,
                item["prompt"],
                n_samples,
                HUMANEVAL_TEMPERATURE,
            )

            n_got = len(completions)
            n_pass = 0
            for completion in completions:
                code = _extract_full_code(item["prompt"], completion, item["entry_point"])
                full_code = code + "\n\n" + item["test"] + f"\ncheck({item['entry_point']})\n"
                if _execute_humaneval(full_code):
                    n_pass += 1

            pass_rate = n_pass / n_got if n_got > 0 else 0.0
            row = {
                "model": model.name,
                "task_id": item["task_id"],
                "entry_point": item["entry_point"],
                "n_samples": n_got,
                "n_pass": n_pass,
                "pass_rate": pass_rate,
            }
            for k in ks:
                row[f"pass@{k}"] = _estimate_pass_at_k(n_got, n_pass, k)
            records.append(row)

            if (idx + 1) % 10 == 0 or idx == 0:
                model_so_far = [r for r in records if r["model"] == model.name]
                parts = [f"[{idx+1}/{len(questions)}]"]
                for k in ks:
                    avg = sum(r[f"pass@{k}"] for r in model_so_far) / len(model_so_far)
                    parts.append(f"pass@{k}={avg:.3f}")
                avg_rate = sum(r["pass_rate"] for r in model_so_far) / len(model_so_far)
                parts.append(f"pass_rate={avg_rate:.3f}")
                print("  " + "  ".join(parts))

            time.sleep(0.5)

        model_records = [r for r in records if r["model"] == model.name]
        n = len(model_records)
        avg_rate = sum(r["pass_rate"] for r in model_records) / n
        print(f"\n  Final [{model}]:")
        for k in ks:
            avg = sum(r[f"pass@{k}"] for r in model_records) / n
            print(f"    pass@{k} = {avg:.3f}")
        print(f"    pass_rate (avg completions passing) = {avg_rate:.3f}")

    return pd.DataFrame(records)


### ── Main ────────────────────────────────────────────────────────────────────

def main():
    # Load all datasets upfront
    datasets = {}
    for name, config in BENCHMARKS.items():
        print(f"Loading {name} dataset...")
        if "loader_fn" in config:
            datasets[name] = config["loader_fn"]()
        else:
            datasets[name] = load_dataset(config["dataset_name"], **config["dataset_kwargs"])
        print(f"  {len(datasets[name])} questions loaded")

    # Run each benchmark
    for name, config in BENCHMARKS.items():
        print(f"\n{'='*60}")
        print(f"Benchmark: {name.upper()}")
        print(f"{'='*60}")

        print(f"Sampling {TOTAL_QUESTIONS} questions...")
        questions = config["sample_fn"](datasets[name], TOTAL_QUESTIONS, RANDOM_SEED)

        df = config["run_fn"](MODELS, questions)

        df.to_csv(config['output_path']+'_results.csv', index=False)
        print(f"\nResults saved to {config['output_path']+'_results.csv'}")


if __name__ == "__main__":
    main()
    
