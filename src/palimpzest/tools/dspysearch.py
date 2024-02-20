import os

import dspy

from palimpzest.tools.dspyadaptors import TogetherHFAdaptor


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

def run_cot_bool(context, question, llmService="openai", verbose=False, promptSignature=FilterOverPaper):
    if llmService == "openai":
        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        # get openai key from environment
        openai_key = os.environ['OPENAI_API_KEY']
        turbo = dspy.OpenAI(model='gpt-4-0125-preview', api_key=openai_key, temperature=0.0)
    elif llmService == "together":
        if 'TOGETHER_API_KEY' not in os.environ:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")
        # get together key from environment
        together_key = os.environ['TOGETHER_API_KEY']
        #redpajamaModel = 'togethercomputer/RedPajama-INCITE-7B-Base'
        mixtralModel = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        turbo = TogetherHFAdaptor(mixtralModel, together_key)
    else:
        raise ValueError("llmService must be either 'openai' or 'together'")

    dspy.settings.configure(lm=turbo)
    cot = dspyCOT(promptSignature)
    pred = cot(question, context)
    if verbose:
        print("Prompt history:")
        turbo.inspect_history(n=1)
    #print(question)
    #print(indent(pred.rationale, 4 * ' '))
    #print(pred.answer)
    return pred.answer

def run_cot_qa(context, question, llmService="openai", verbose=False, promptSignature=QuestionOverPaper):
    if llmService == "openai":
        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        # get openai key from environment
        openai_key = os.environ['OPENAI_API_KEY']
        turbo = dspy.OpenAI(model='gpt-4-0125-preview', api_key=openai_key, temperature=0.0)
    elif llmService == "together":
        if 'TOGETHER_API_KEY' not in os.environ:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")
        # get together key from environment
        together_key = os.environ['TOGETHER_API_KEY']
        #redpajamaModel = 'togethercomputer/RedPajama-INCITE-7B-Base'
        mixtralModel = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        turbo = TogetherHFAdaptor(mixtralModel, together_key)
    else:
        raise ValueError("llmService must be either 'openai' or 'together'")

    dspy.settings.configure(lm=turbo)
    cot = dspyCOT(promptSignature)
    pred = cot(question, context)
    if verbose:
        print("Prompt history:")
        turbo.inspect_history(n=1)
    return pred.answer

if __name__ == "__main__":
    llmService = "openai"
    if llmService == "openai":
        if 'OPENAI_API_KEY' not in os.environ:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        # get openai key from environment
        openai_key = os.environ['OPENAI_API_KEY']
        turbo = dspy.OpenAI(model='gpt-4-0125-preview', api_key=openai_key, temperature=0.0)
    elif llmService == "together":
        if 'TOGETHER_API_KEY' not in os.environ:
            raise ValueError("TOGETHER_API_KEY not found in environment variables")
        # get together key from environment
        together_key = os.environ['TOGETHER_API_KEY']
        #redpajamaModel = 'togethercomputer/RedPajama-INCITE-7B-Base'
        mixtralModel = 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        turbo = TogetherHFAdaptor(mixtralModel, together_key)
    else:
        raise ValueError("llmService must be either 'openai' or 'together'")

    dspy.settings.configure(lm=turbo)

    cot = dspyCOT(QuestionOverPaper)
    question = """What is the title of the paper?
    Who is the first author?
    What is the first author's institution?"""
    context = open("../../../tests/testFileDirectory/cosmos/1_All_F_Guo/1 All F Guo.txt").read()
    pred = cot(question, context)
