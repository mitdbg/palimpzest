import os
from textwrap import indent

import dspy

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

class RAG(dspy.Module):
    def __init__(self, f_signature=FilterOverPaper):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(f_signature)

    def forward(self, question, context):
        context = context
        answer = self.generate_answer(context=context, question=question)
        return answer

def run_rag_boolean(context, question):
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    # get openai key from environment
    openai_key = os.environ['OPENAI_API_KEY']
    turbo = dspy.OpenAI(model='gpt-4-0125-preview', api_key=openai_key, temperature=0.0)
    dspy.settings.configure(lm=turbo)
    rag = RAG(FilterOverPaper)
    pred = rag(question, context)
    print(question)
    print(indent(pred.rationale, 4 * ' '))
    print(pred.answer)
    return pred.answer

def run_rag_qa(context, question):
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    # get openai key from environment
    openai_key = os.environ['OPENAI_API_KEY']
    turbo = dspy.OpenAI(model='gpt-4-0125-preview', api_key=openai_key, temperature=0.0)
    dspy.settings.configure(lm=turbo)
    rag = RAG(QuestionOverPaper)
    pred = rag(question, context)
    print(question)
    print(indent(pred.rationale, 4 * ' '))
    print(pred.answer)
    return pred.answer

if __name__ == "__main__":
    # get openai key from environment, throw error if not found
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    # get openai key from environment
    openai_key = os.environ['OPENAI_API_KEY']

    turbo = dspy.OpenAI(model='gpt-4-0125-preview', api_key=openai_key, temperature=0.0)
    dspy.settings.configure(lm=turbo)

    # rag = RAG(FilterOverPaper)
    # question = "Is the paper about batteries?"
    # context = open("../../../tests/testFileDirectory/cosmos/1_All_F_Guo/1 All F Guo.txt").read()
    # pred = rag(question, context)
    # print(question)
    # print(indent(pred.rationale, 4 * ' '))
    # print(pred.answer)
    # print()
    # question = "Is the paper from MIT?"
    # pred = rag(question, context)
    # print(question)
    # print(indent(pred.rationale, 4 * ' '))
    # print(pred.answer)
    # print()

    rag = RAG(QuestionOverPaper)
    question = """What is the title of the paper?
    Who is the first author?
    What is the first author's institution?"""
    context = open("../../../tests/testFileDirectory/cosmos/1_All_F_Guo/1 All F Guo.txt").read()
    pred = rag(question, context)
    print(question)
    print(indent(pred.rationale, 4 * ' '))
    print(pred.answer)

