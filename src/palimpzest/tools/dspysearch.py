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

class RAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(FilterOverPaper)

    def forward(self, question, context):
        context = context
        answer = self.generate_answer(context=context, question=question)
        return answer

def run_rag(context, question):
    if 'OPENAI_API_KEY' not in os.environ:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    # get openai key from environment
    openai_key = os.environ['OPENAI_API_KEY']
    turbo = dspy.OpenAI(model='gpt-4-0125-preview', api_key=openai_key, temperature=0.0)
    dspy.settings.configure(lm=turbo)
    rag = RAG()
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

    rag = RAG()
    question = "Is the paper about batteries?"
    context = open("../../../tests/testFileDirectory/cosmos/1_All_F_Guo/1 All F Guo.txt").read()
    pred = rag(question, context)
    print(question)
    print(indent(pred.rationale, 4 * ' '))
    print(pred.answer)
    print()
    question = "Is the paper from MIT?"
    pred = rag(question, context)
    print(question)
    print(indent(pred.rationale, 4 * ' '))
    print(pred.answer)