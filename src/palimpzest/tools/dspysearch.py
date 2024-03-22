from palimpzest.constants import Model
from palimpzest.tools.dspyadaptors import TogetherHFAdaptor
from palimpzest.tools.profiler import Profiler

import dspy
import os
import time


##
# Given a question, we'll feed it with the paper context for answer generation.
##
class FilterOverPaper(dspy.Signature):
    """Answer condition questions about a scientific paper."""

    context = dspy.InputField(desc="contains full text of the paper, including author, institution, title, and body")
    question = dspy.InputField(desc="one or more conditions about the paper")
    answer = dspy.OutputField(desc="often a TRUE/FALSE answer to the condition question(s) about the paper")


class QuestionOverPaper(dspy.Signature):
    """Answer question(s) about a scientific paper."""

    context = dspy.InputField(desc="contains full text of the paper, including author, institution, title, and body")
    question = dspy.InputField(desc="one or more question about the paper")
    answer = dspy.OutputField(desc="print the answer only, separated by a newline character")


#invoke dspy in chain of thought mode
class dspyCOT(dspy.Module):
    def __init__(self, f_signature=FilterOverPaper):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(f_signature)

    def forward(self, question, context):
        context = context
        answer = self.generate_answer(context=context, question=question)
        return answer


def gen_signature_class(instruction, context_desc, question_desc, answer_desc):
    class QuestionOverDoc(dspy.Signature):
        __doc__ = instruction
        context = dspy.InputField(desc= context_desc)
        question = dspy.InputField(desc= question_desc)
        answer = dspy.OutputField(desc= answer_desc)
    return QuestionOverDoc

def gen_filter_signature_class(doc_schema, doc_type):
    instruction = f"Answer condition questions about a {doc_schema}."
    context_desc = f"contains full text of the {doc_type}"
    question_desc = f"one or more conditions about the {doc_type}"
    answer_desc = f"often a TRUE/FALSE answer to the condition question(s) about the {doc_type}"
    return gen_signature_class(instruction, context_desc, question_desc, answer_desc)

def gen_qa_signature_class(doc_schema, doc_type):
    instruction = f"Answer question(s) about a {doc_schema}."
    context_desc = f"contains full text of the {doc_type}"
    question_desc = f"one or more question about the {doc_type}"
    answer_desc = f"print the answer only, separated by a newline character"
    return gen_signature_class(instruction, context_desc, question_desc, answer_desc)

def run_cot_bool(context, question, model_name, verbose=False, promptSignature=FilterOverPaper):
    if model_name in [Model.GPT_3_5.value, Model.GPT_4.value]:
        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        # get openai key from environment
        openai_key = os.environ['OPENAI_API_KEY']
        turbo = dspy.OpenAI(model=model_name, api_key=openai_key, temperature=0.0)
    elif model_name in [Model.MIXTRAL.value]:
        if 'TOGETHER_API_KEY' not in os.environ:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")
        # get together key from environment
        together_key = os.environ['TOGETHER_API_KEY']
        #redpajamaModel = 'togethercomputer/RedPajama-INCITE-7B-Base'
        # mixtralModel = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        mixtralModel = model_name
        turbo = TogetherHFAdaptor(mixtralModel, together_key)
    elif model_name in [Model.GEMINI_1.value, Model.GEMINI_1V.value]:
        if 'GOOGLE_API_KEY' not in os.environ:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        # get google key from environment
        google_key = os.environ['GOOGLE_API_KEY']
        turbo = dspy.Google(model=model_name, api_key=google_key)
    else:
        raise ValueError("model must be one of those specified in palimpzest.constants.Model")

    dspy.settings.configure(lm=turbo)
    cot = dspyCOT(promptSignature)

    start_time = time.time()
    pred = cot(question, context)
    end_time = time.time()

    # TODO: need to create some class structure / abstraction around everything from
    #       physical operators -> solvers -> dspysearch, dspyadaptors, openai_image_converter, etc.
    # collect statistics on prompt, usage, and timing if profiling is on
    stats = {}
    if Profiler.profiling_on():
        stats['api_call_duration'] = end_time - start_time
        stats['prompt'] = turbo.history[-1]['prompt']
        stats['usage'] = turbo.history[-1]['response']['usage']
        stats['finish_reason'] = (
            turbo.history[-1]['response']['finish_reason']
            if isinstance(turbo, TogetherHFAdaptor)
            else turbo.history[-1]['response']['choices'][-1]['finish_reason']
        )

    if verbose:
        print("Prompt history:")
        turbo.inspect_history(n=1)

    return pred.answer, stats


def run_cot_qa(context, question, model_name, verbose=False, promptSignature=QuestionOverPaper):
    if model_name in [Model.GPT_3_5.value, Model.GPT_4.value]:
        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        # get openai key from environment
        openai_key = os.environ['OPENAI_API_KEY']
        turbo = dspy.OpenAI(model=model_name, api_key=openai_key, temperature=0.0, max_tokens=4096)
    elif model_name in [Model.MIXTRAL.value]:
        if 'TOGETHER_API_KEY' not in os.environ:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")
        # get together key from environment
        together_key = os.environ['TOGETHER_API_KEY']
        #redpajamaModel = 'togethercomputer/RedPajama-INCITE-7B-Base'
        # mixtralModel = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        mixtralModel = model_name
        turbo = TogetherHFAdaptor(mixtralModel, together_key)
    elif model_name in [Model.GEMINI_1.value, Model.GEMINI_1V.value]:
        if 'GOOGLE_API_KEY' not in os.environ:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        # get google key from environment
        google_key = os.environ['GOOGLE_API_KEY']
        turbo = dspy.Google(model=model_name, api_key=google_key)
    else:
        raise ValueError("model must be one of those specified in palimpzest.constants.Model")

    dspy.settings.configure(lm=turbo)
    cot = dspyCOT(promptSignature)

    start_time = time.time()
    pred = cot(question, context)
    end_time = time.time()

    # TODO: need to create some class structure / abstraction around everything from
    #       physical operators -> solvers -> dspysearch, dspyadaptors, openai_image_converter, etc.
    # collect statistics on prompt, usage, and timing if profiling is on
    stats = {}
    if Profiler.profiling_on():
        stats['api_call_duration'] = end_time - start_time
        stats['prompt'] = turbo.history[-1]['prompt']
        stats['usage'] = turbo.history[-1]['response']['usage']
        stats['finish_reason'] = (
            turbo.history[-1]['response']['finish_reason']
            if isinstance(turbo, TogetherHFAdaptor)
            else turbo.history[-1]['response']['choices'][-1]['finish_reason']
        )

    if verbose:
        print("Prompt history:")
        turbo.inspect_history(n=1)

    return pred.answer, stats

