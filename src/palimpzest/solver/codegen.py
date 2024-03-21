from pyheaven import *
from palimpzest.constants import Model
from palimpzest.solver.sandbox import API
from palimpzest.elements import Schema, Field
from palimpzest.datamanager import DataDirectory
from palimpzest.tools.dspysearch import run_codegen, run_basic_request, exec_codegen
from typing import Any, Dict, Tuple, Union

# TODO: support parallel (edit cache)

CODEGEN_PROMPT = """You are a helpful programming assistant and an expert Python programmer. Implement the Python function `{api}` that extracts `{output}` ({output_desc}) from given inputs:
{inputs_desc}
Example:
{example_inputs}
{example_output}
Notice that the evaluation will severely punish incorrect outputs. Thus, when the function is uncertain, it should return `None` to abstain instead of returning an incorrect guess.
Return the implementation only."""

NONE_CODE = """
def {api}(contents, filename):
    return None
"""

def llmCodeGen(prompt, model, default=NONE_CODE.format(api='extract')):
    try:
        code, _ = run_codegen(prompt, model_name=model.value); return code
    except Exception as e:
        return default


ADVICEGEN_PROMPT = """You are a helpful programming assistant and an expert Python programmer. Your job is to provide programming ideas to help me write Python programs.
For example, if I want to complete a task: "extract the salary number (in USD) from a given employee's document", you can provide me with {n} different ways to do it like:
Idea 1: Use regular expressions to extract the salary number: a number with a dollar sign in front of it. For example, $100,000.
Idea 2: Find the table entry with the salary number.
Idea 3: Use a pre-trained NLP model to extract the salary number.

Now, consider the following Python programming task that extracts `{output}` ({output_desc}) from given inputs:
{inputs_desc}
Example:
{example_inputs}
{example_output}
Please provide me with {n} different ideas to complete this task. Return the ideas only, following the format above.
"""
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

def llmAdviceGen(prompt, model, default=list()):
    try:
        response, _ = run_basic_request(prompt, model_name=model.value)
        advices = list(parse_ideas(response).values()); return advices
    except Exception as e:
        return default

ADVICED_CODEGEN_PROMPT = """You are a helpful programming assistant and an expert Python programmer. Implement the Python function `{api}` that extracts `{output}` ({output_desc}) from given inputs:
{inputs_desc}
Example:
{example_inputs}
{example_output}
Hint: {advice}
Notice that the evaluation will severely punish incorrect outputs. Thus, when the function is uncertain, it should return `None` to abstain instead of returning an incorrect guess.
Return the implementation only."""



def getConversionCodeExamples(inputSchema: Schema, outputFieldDict: Dict[str, Field], conversionDesc: str, config: Dict[str, Any]):
    examplesdir = pjoin(DataDirectory()._dir, "data", "cache", "codegen_cache_examples.json")
    if not ExistFile(examplesdir): SaveJson(dict(), examplesdir)
    EXAMPLES_CACHE = LoadJson(examplesdir)
    outputName, outputField = list(outputFieldDict.items())[0]
    hash_key = str((conversionDesc, inputSchema, outputName, outputField).__hash__())
    if hash_key in EXAMPLES_CACHE:
        return EXAMPLES_CACHE[hash_key]
    return list()

def registerConversionCodeGenExample(inputSchema: Schema, outputFieldDict: Dict[str, Field], conversionDesc: str, inputs: Dict[str, Any], outputs: Dict[str, Any]=None):
    examplesdir = pjoin(DataDirectory()._dir, "data", "cache", "codegen_cache_examples.json")
    if not ExistFile(examplesdir): SaveJson(dict(), examplesdir)
    EXAMPLES_CACHE = LoadJson(examplesdir)
    outputName, outputField = list(outputFieldDict.items())[0]
    hash_key = str((conversionDesc, inputSchema, outputName, outputField).__hash__())
    if hash_key not in EXAMPLES_CACHE:
        EXAMPLES_CACHE[hash_key] = list()
    EXAMPLES_CACHE[hash_key].append((inputs, outputs))
    SaveJson(EXAMPLES_CACHE, examplesdir)

def codeGen(inputSchema: Schema, outputFieldDict: Dict[str, Field], conversionDesc: str, config: Dict[str, Any], model: Model, api: API):
    outputName, outputField = list(outputFieldDict.items())[0]
    # LLM code generation
    examples = getConversionCodeExamples(inputSchema, outputFieldDict, conversionDesc, config)
    assert (len(examples) > 0), "No examples found for code generation"
    example_inputs, example_output = examples[0] # TODO: use more examples
    codeGenPrompt = CODEGEN_PROMPT.format(
        api = api.args_call(),
        output = outputName,
        output_desc = f"{outputField.desc}. {conversionDesc}" if conversionDesc else outputField.desc,
        inputs_desc = "\n".join([f"- {k}: {getattr(inputSchema,k).desc}." for k in inputSchema.fieldNames()]),
        example_inputs = "\n".join([f"- {k} = {repr(example_inputs[k])}" for k in inputSchema.fieldNames()]),
        example_output = f"- {outputName} = {repr(example_output[outputName])}" if example_output else ""
    )
    code = llmCodeGen(codeGenPrompt, model, default=NONE_CODE.format(api=api.name))
    
    # more complex code generation process
    num_ensemble        =  config.get('codegen_num_ensemble',       default=    4)
    validation          =  config.get('codegen_validation',         default=False)
    num_iterations      =  config.get('codegen_num_iterations',     default=    5)
    num_max_examples    =  config.get('codegen_num_max_examples',   default=   20)
    num_advices = max(num_ensemble-1, 0)
    codes = [code]
    if num_advices > 0:
        # Advised Code Generation
        adviceGenPrompt = ADVICEGEN_PROMPT.format(
            api = api.args_call(),
            output = outputName,
            output_desc = f"{outputField.desc}. {conversionDesc}" if conversionDesc else outputField.desc,
            inputs_desc = "\n".join([f"- {k}: {getattr(inputSchema,k).desc}." for k in inputSchema.fieldNames()]),
            example_inputs = "\n".join([f"- {k} = {repr(example_inputs[k])}" for k in inputSchema.fieldNames()]),
            example_output = f"- {outputName} = {repr(example_output[outputName])}" if example_output else "",
            n = num_advices,
        )
        advices = llmAdviceGen(adviceGenPrompt, model)
        for advice in advices:
            advisedCodeGenPrompt = ADVICED_CODEGEN_PROMPT.format(
                api = api.args_call(),
                output = outputName,
                output_desc = f"{outputField.desc}. {conversionDesc}" if conversionDesc else outputField.desc,
                inputs_desc = "\n".join([f"- {k}: {getattr(inputSchema,k).desc}." for k in inputSchema.fieldNames()]),
                example_inputs = "\n".join([f"- {k} = {repr(example_inputs[k])}" for k in inputSchema.fieldNames()]),
                example_output = f"- {outputName} = {repr(example_output[outputName])}" if example_output else "",
                advice = advice,
            )
            advised_code = llmCodeGen(advisedCodeGenPrompt, model, default=NONE_CODE.format(api=api.name))
            codes.append(advised_code)
    
        # Validation & Iteration (there is currently not a lot of validation examples, and the reGenerate is always False, so this may never be used)
        if validation and num_iterations > 0:
            validation_examples = [(example_inputs, example_output) for example_inputs, example_output in validation_examples if example_output is not None]
            validation_examples = validation_examples[:num_max_examples]
            profiles = [[(exec_codegen(api, code, example_inputs), example_output) for exp in validation_examples] for code in codes]
            for i in range(num_iterations):
                # TODO: need validation examples, probably from a forced fallback ratio to LLM (e.g., 1% ?)
                pass

    return codes

def getConversionCodes(inputSchema: Schema, outputFieldDict: Dict[str, Field], conversionDesc: str, config: Dict[str, Any], model: Model, api: API, reGenerate: bool=False):
    cachedir = pjoin(DataDirectory()._dir, "data", "cache", "codegen_cache_code.json")
    if not ExistFile(cachedir): SaveJson(dict(), cachedir)
    CODEGEN_CACHE = LoadJson(cachedir)
    outputName, outputField = list(outputFieldDict.items())[0]
    hash_key = str((conversionDesc, inputSchema, outputName, outputField).__hash__())
    if (hash_key in CODEGEN_CACHE) and (not reGenerate):
        return CODEGEN_CACHE[hash_key]
    codes = codeGen(inputSchema, outputFieldDict, conversionDesc, config, model, api)
    CODEGEN_CACHE[hash_key] = codes
    SaveJson(CODEGEN_CACHE, cachedir)
    return codes