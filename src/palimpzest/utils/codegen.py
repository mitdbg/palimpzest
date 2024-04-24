



from palimpzest.profiler import Stats, GenerationStats, CodeGenSingleStats, CodeGenEnsembleStats, CodeExecutionSingleStats, CodeExecutionEnsembleStats, FullCodeGenStats
from palimpzest.utils import API
from palimpzest.constants import Model, CodeGenStrategy
from palimpzest.elements import DataRecord

from typing import Any, Dict, List, Tuple
from collections import Counter

import time

from seed import LLM
turbo = LLM()
def _run_basic_request(prompt, model_name):
    start_time = time.time()
    response = turbo.q([{'role': 'user', 'content': prompt}])
    pred = response['text'] if response['status'] and response['text'] else None
    end_time = time.time()
    
    stats = GenerationStats(
        model_name = model_name,
        llm_call_duration_secs = end_time - start_time,
        prompt = prompt,
        # ...
    )
    return pred, stats

def run_codegen(prompt, model_name=Model.GPT_4.value, language='Python'):
    pred, stats = _run_basic_request(prompt, model_name)
    ordered_keys = [
        f'```{language}',
        f'```{language.lower()}',
        f'```'
    ]
    code = None
    for key in ordered_keys:
        if key in pred:
            code = pred.split(key)[1].split('```')[0].strip()
            break
    return code, stats

def parse_multiple_outputs(text, outputs=['Thought', 'Action']):
    data = {}
    for key in reversed(outputs):
        if key+':' in text:
            remain, value = text.rsplit(key+':', 1)
            data[key.lower()] = value.strip()
            text = remain
        else:
            data[key.lower()] = None
    return data

def parse_ideas(text, limit=3):
    return parse_multiple_outputs(text, outputs=[f'Idea {i}' for i in range(1, limit+1)])

def run_advgen(prompt, model_name=Model.GPT_4.value):
    pred, stats = _run_basic_request(prompt, model_name)
    advs = parse_ideas(pred); return advs, stats

def codeGenDefault(api):
    return api.api_def()+"  return None\n", GenerationStats()

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
def codeGenSingle(api: API, examples: List[Dict[DataRecord, DataRecord]]=list(), advice: str=None, model_name=Model.GPT_4.value, language='Python'):
    prompt_template = CODEGEN_PROMPT
    context = {
        'language': language,
        'api': api.args_call(),
        'output': api.output,
        'output_desc': api.output_desc,
        'examples_desc': "\n".join([
            EXAMPLE_PROMPT.format(
                idx = f" {i}" if len(examples)>1 else "",
                example_inputs = "\n".join([f"- {k} = {repr(getattr(example, k))}" for k in api.inputs]),
                example_output = ""
            ) for i, example in enumerate(examples, 1)
        ]),
        'advice': f"Hint: {advice}" if advice else "",
    }
    prompt = prompt_template.format(**context)
    code, gen_stats = run_codegen(prompt, model_name)
    stats = CodeGenSingleStats(
        prompt_template = prompt_template,
        context = context,
        code = code,
        gen_stats = gen_stats,
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
def adviceGen(api: API, examples: List[Dict[DataRecord, DataRecord]]=list(), model_name=Model.GPT_4.value, language='Python', n_advices=4):
    prompt_template = ADVICEGEN_PROMPT
    context = {
        'language': language,
        'api': api.args_call(),
        'output': api.output,
        'output_desc': api.output_desc,
        'examples_desc': "\n".join([
            EXAMPLE_PROMPT.format(
                idx = f" {i}" if len(examples)>1 else "",
                example_inputs = "\n".join([f"- {k} = {repr(getattr(example, k))}" for k in api.inputs]),
                example_output = ""
            ) for i, example in enumerate(examples, 1)
        ]),
        'n': n_advices,
    }
    prompt = prompt_template.format(**context)
    advs, stats = run_advgen(prompt, model_name)
    return advs, stats

def reGenerationCondition(api: API, examples: List[Dict[DataRecord, DataRecord]]=list(), strategy: CodeGenStrategy=CodeGenStrategy.DEFAULT,
    code_ensemble: int=4,               # if strategy != SINGLE
    code_num_examples: int=1,           # if strategy != EXAMPLE_ENSEMBLE
    code_regenerate_frequency: int=200, # if strategy == ADVICE_ENSEMBLE_WITH_VALIDATION
) -> bool:
    if strategy == CodeGenStrategy.NONE:
        return False
    if strategy == CodeGenStrategy.EXAMPLE_ENSEMBLE:
        return len(examples) <= code_ensemble
    if strategy == CodeGenStrategy.ADVICE_ENSEMBLE:
        return False
    if strategy == CodeGenStrategy.ADVICE_ENSEMBLE_WITH_VALIDATION:
        return len(examples)%code_regenerate_frequency == 0

def codeEnsembleGeneration(api: API, examples: List[Dict[DataRecord, DataRecord]]=list(), strategy: CodeGenStrategy=CodeGenStrategy.DEFAULT,
    code_ensemble: int=4,               # if strategy != SINGLE
    code_num_examples: int=1,           # if strategy != EXAMPLE_ENSEMBLE
    code_regenerate_frequency: int=200, # if strategy == ADVICE_ENSEMBLE_WITH_VALIDATION
) -> Tuple[Dict[str, str], CodeGenEnsembleStats]:
    code_ensemble = dict(); code_gen_stats = CodeGenEnsembleStats()
    if strategy == CodeGenStrategy.NONE:
        code, stats = codeGenDefault(api)
        for i in range(code_ensemble):
            code_name = f"{api.name}_v{i}"
            code_gen_stats.code_versions_stats[code_name] = stats
            code_ensemble[code_name] = code
        return code_ensemble, code_gen_stats
    if strategy == CodeGenStrategy.SINGLE:
        code, stats = codeGenSingle(api, examples=examples[:code_num_examples])
        for i in range(code_ensemble):
            code_name = f"{api.name}_v{i}"
            code_gen_stats.code_versions_stats[code_name] = stats
            code_ensemble[code_name] = code
        return code_ensemble, code_gen_stats
    if strategy == CodeGenStrategy.EXAMPLE_ENSEMBLE:
        for i in range(code_ensemble):
            code_name = f"{api.name}_v{i}"
            code, stats = codeGenSingle(api, examples=[examples[i]])
            code_gen_stats.code_versions_stats[code_name] = stats
            code_ensemble[code_name] = code
        return code_ensemble, code_gen_stats
    if strategy == CodeGenStrategy.ADVICE_ENSEMBLE:
        advices, adv_stats = adviceGen(api, examples=examples[:code_num_examples])
        code_gen_stats.advice_gen_stats = adv_stats
        for i, adv in enumerate(advices):
            code_name = f"{api.name}_v{i}"
            code, stats = codeGenSingle(api, examples=examples[:code_num_examples], advice=adv)
            code_gen_stats.code_versions_stats[code_name] = stats
            code_ensemble[code_name] = code
        return code_ensemble, code_gen_stats
    if strategy == CodeGenStrategy.ADVICE_ENSEMBLE_WITH_VALIDATION:
        raise Exception("not implemented yet")

def codeExecution(api: API, code: str, candidate: DataRecord, verbose:bool=False):
    start_time = time.time()
    inputs = {field_name:getattr(candidate,field_name) for field_name in api.inputs}
    response = api.api_execute(code, inputs)
    pred = response['response'] if response['status'] and response['response'] else None
    end_time = time.time()
    stats = CodeExecutionSingleStats(
        code_response = response,
        code_exec_duration_secs = end_time - start_time,
    )
    return pred, stats

# Temporarily set default verbose to True for debugging
def codeEnsembleExecution(api: API, code_ensemble: List[str], candidate: DataRecord, verbose:bool=True) -> Tuple[DataRecord, Dict]:
    ensemble_stats = CodeExecutionEnsembleStats(); preds = list()
    for code_name, code in code_ensemble.items():
        pred, stats = codeExecution(api, code, candidate)
        preds.append(pred)
        ensemble_stats.code_versions_stats[code_name] = stats
    preds = [pred for pred in preds if pred is not None]
    if len(preds) > 0:
        majority_response = Counter(preds).most_common(1)[0][0]
        ensemble_stats.majority_response = majority_response
        return majority_response+(" (codegen)" if verbose else ""), ensemble_stats
    return None, ensemble_stats
