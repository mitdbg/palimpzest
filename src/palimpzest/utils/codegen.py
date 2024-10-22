import time
from collections import Counter
from typing import Any, Dict, List, Tuple

from palimpzest.constants import CodeGenStrategy, Model
from palimpzest.elements import DataRecord
from palimpzest.generators import CustomGenerator

from .sandbox import API

llm = CustomGenerator(model_name=Model.GPT_4.value)


def run_codegen(prompt, language="Python"):
    pred, stats = llm.generate(prompt=prompt)
    ordered_keys = [f"```{language}", f"```{language.lower()}", "```"]
    code = None
    for key in ordered_keys:
        if key in pred:
            code = pred.split(key)[1].split("```")[0].strip()
            break
    return code, stats


def parse_multiple_outputs(text, outputs=["Thought", "Action"]):
    data = {}
    for key in reversed(outputs):
        if key + ":" in text:
            remain, value = text.rsplit(key + ":", 1)
            data[key.lower()] = value.strip()
            text = remain
        else:
            data[key.lower()] = None
    return data


def parse_ideas(text, limit=3):
    return parse_multiple_outputs(text, outputs=[f"Idea {i}" for i in range(1, limit + 1)])


def run_advgen(prompt):
    pred, stats = llm.generate(prompt=prompt)
    advs = parse_ideas(pred)
    return advs, stats


def codeGenDefault(api):
    return api.api_def() + "  return None\n", GenerationStats()


EXAMPLE_PROMPT = """Example{idx}:
{example_inputs}
{example_output}
"""
CODEGEN_PROMPT = """You are a helpful programming assistant and an expert {language} programmer. Implement the {language} function `{api}` that extracts `{output}` ({output_desc}) from given inputs:
{inputs_desc}
{examples_desc}
Notice that the evaluation will severely punish incorrect outputs. Thus, when the function is uncertain, it should return `None` to abstain instead of returning an incorrect guess.
{advice}
Return the implementation only."""


# NOTE: I think examples was List[DataRecord] and is now List[dict]
def codeGenSingle(
    api: API, examples: List[Dict[DataRecord, DataRecord]] = list(), advice: str = None, language="Python"
):
    prompt_template = CODEGEN_PROMPT
    context = {
        "language": language,
        "api": api.args_call(),
        "output": api.output,
        "inputs_desc": "\n".join([f"- {k} ({api.input_descs[i]})" for i, k in enumerate(api.inputs)]),
        "output_desc": api.output_desc,
        "examples_desc": "\n".join(
            [
                EXAMPLE_PROMPT.format(
                    idx=f" {i}" if len(examples) > 1 else "",
                    example_inputs="\n".join([f"- {k} = {repr(example[k])}" for k in api.inputs]),
                    example_output="",
                )
                for i, example in enumerate(examples, 1)
            ]
        ),
        "advice": f"Hint: {advice}" if advice else "",
    }
    prompt = prompt_template.format(**context)
    print("PROMPT")
    print("-------")
    print(f"{prompt}")
    code, gen_stats = run_codegen(prompt, language=language)
    print("-------")
    print("GENERATED CODE")
    print("---------------")
    print(f"{code}")
    stats = CodeGenSingleStats(
        prompt_template=prompt_template,
        context=context,
        code=code,
        gen_stats=gen_stats,
    )
    return code, stats


ADVICEGEN_PROMPT = """You are a helpful programming assistant and an expert {language} programmer. Your job is to provide programming ideas to help me write {language} programs.
For example, if I want to complete a task: "extract the salary number (in USD) from a given employee's document", you can provide me with {n} different ways to do it like:
Idea 1: Use regular expressions to extract the salary number: a number with a dollar sign in front of it. For example, $100,000.
Idea 2: Find the table entry with the salary number.
Idea 3: Use a pre-trained NLP model to extract the salary number.
# 
Now, consider the following {language} programming task that extracts `{output}` ({output_desc}) from given inputs:
{examples_desc}
Please provide me with {n} different ideas to complete this task. Return the ideas only, following the format above.
"""


# NOTE: I think examples was List[DataRecord] and is now List[dict]
def adviceGen(api: API, examples: List[Dict[DataRecord, DataRecord]] = list(), language="Python", n_advices=4):
    prompt_template = ADVICEGEN_PROMPT
    context = {
        "language": language,
        "api": api.args_call(),
        "output": api.output,
        "inputs_desc": "\n".join([f"- {k} ({api.input_descs[i]})" for i, k in enumerate(api.inputs)]),
        "output_desc": api.output_desc,
        "examples_desc": "\n".join(
            [
                EXAMPLE_PROMPT.format(
                    idx=f" {i}" if len(examples) > 1 else "",
                    example_inputs="\n".join([f"- {k} = {repr(example[k])}" for k in api.inputs]),
                    example_output="",
                )
                for i, example in enumerate(examples, 1)
            ]
        ),
        "n": n_advices,
    }
    prompt = prompt_template.format(**context)
    advs, stats = run_advgen(prompt)
    return advs, stats


# NOTE: I think examples was List[DataRecord] and is now List[dict]
def reGenerationCondition(
    api: API,
    examples: List[Dict[DataRecord, DataRecord]] = list(),
    strategy: CodeGenStrategy = CodeGenStrategy.SINGLE,
    code_ensemble: int = 4,  # if strategy != SINGLE
    code_num_examples: int = 1,  # if strategy != EXAMPLE_ENSEMBLE
    code_regenerate_frequency: int = 200,  # if strategy == ADVICE_ENSEMBLE_WITH_VALIDATION
) -> bool:
    if strategy == CodeGenStrategy.NONE:
        return False
    if strategy == CodeGenStrategy.EXAMPLE_ENSEMBLE:
        return len(examples) <= code_ensemble
    if strategy == CodeGenStrategy.ADVICE_ENSEMBLE:
        return False
    if strategy == CodeGenStrategy.ADVICE_ENSEMBLE_WITH_VALIDATION:
        return len(examples) % code_regenerate_frequency == 0


# NOTE: I think examples was List[DataRecord] and is now List[dict]
def codeEnsembleGeneration(
    api: API,
    examples: List[Dict[DataRecord, DataRecord]] = list(),
    strategy: CodeGenStrategy = CodeGenStrategy.SINGLE,
    code_ensemble_num: int = 1,  # if strategy != SINGLE
    code_num_examples: int = 1,  # if strategy != EXAMPLE_ENSEMBLE
    code_regenerate_frequency: int = 200,  # if strategy == ADVICE_ENSEMBLE_WITH_VALIDATION
) -> Tuple[Dict[str, str], CodeGenEnsembleStats]:
    code_ensemble = dict()
    code_gen_stats = CodeGenEnsembleStats()
    if strategy == CodeGenStrategy.NONE:
        code, stats = codeGenDefault(api)
        for i in range(code_ensemble_num):
            code_name = f"{api.name}_v{i}"
            code_gen_stats.code_versions_stats[code_name] = stats
            code_ensemble[code_name] = code
        return code_ensemble, code_gen_stats
    if strategy == CodeGenStrategy.SINGLE:
        code, stats = codeGenSingle(api, examples=examples[:code_num_examples])
        for i in range(code_ensemble_num):
            code_name = f"{api.name}_v{i}"
            code_gen_stats.code_versions_stats[code_name] = stats
            code_ensemble[code_name] = code
        return code_ensemble, code_gen_stats
    if strategy == CodeGenStrategy.EXAMPLE_ENSEMBLE:
        for i in range(code_ensemble_num):
            code_name = f"{api.name}_v{i}"
            code, stats = codeGenSingle(api, examples=[examples[i]])
            code_gen_stats.code_versions_stats[code_name] = stats
            code_ensemble[code_name] = code
        return code_ensemble, code_gen_stats
    if strategy == CodeGenStrategy.ADVICE_ENSEMBLE:
        advices, adv_stats = adviceGen(api, examples=examples[:code_num_examples], n_advices=code_ensemble_num)
        code_gen_stats.advice_gen_stats = adv_stats
        for i, adv in enumerate(advices):
            code_name = f"{api.name}_v{i}"
            code, stats = codeGenSingle(api, examples=examples[:code_num_examples], advice=adv)
            code_gen_stats.code_versions_stats[code_name] = stats
            code_ensemble[code_name] = code
        return code_ensemble, code_gen_stats
    if strategy == CodeGenStrategy.ADVICE_ENSEMBLE_WITH_VALIDATION:
        raise Exception("not implemented yet")


def codeExecution(api: API, code: str, candidate_dict: Dict[str, Any], verbose: bool = False):
    start_time = time.time()
    inputs = {field_name: candidate_dict[field_name] for field_name in api.inputs}
    response = api.api_execute(code, inputs)
    pred = response["response"] if response["status"] and response["response"] else None
    end_time = time.time()
    stats = CodeExecutionSingleStats(
        code_response=response,
        code_exec_duration_secs=end_time - start_time,
    )
    return pred, stats


# Temporarily set default verbose to True for debugging
def codeEnsembleExecution(
    api: API, code_ensemble: List[str], candidate_dict: Dict[str, Any], verbose: bool = True
) -> Tuple[DataRecord, Dict]:
    ensemble_stats = CodeExecutionEnsembleStats()
    preds = list()
    for code_name, code in code_ensemble.items():
        pred, stats = codeExecution(api, code, candidate_dict)
        preds.append(pred)
        ensemble_stats.code_versions_stats[code_name] = stats
    preds = [pred for pred in preds if pred is not None]
    print(preds)

    # TODO: short-term hack to avoid calling Counter(preds) when preds is a list for biofabric (which is unhashable)
    #
    if len(preds) == 1:
        majority_response = preds[0]
        ensemble_stats.majority_response = majority_response
        return majority_response, ensemble_stats

    if len(preds) > 0:
        majority_response = Counter(preds).most_common(1)[0][0]
        ensemble_stats.majority_response = majority_response
        # return majority_response+(" (codegen)" if verbose else ""), ensemble_stats
        return majority_response, ensemble_stats
    return None, ensemble_stats
