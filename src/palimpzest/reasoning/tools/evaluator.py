import json
from dataclasses import dataclass

from palimpzest.reasoning.action_types import ExecuteAction
from palimpzest.reasoning.llm_api.llm_api import LLMClient
from palimpzest.reasoning.tools.prompt_checker import PromptChecker
from palimpzest.reasoning.eval_response import EvaluationResponse


class Evaluator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.llm_client = LLMClient(model=model)

    def evluator_prompt(self, question: str, answer: str, context: str):
        return {
            "system": "You are an evaluator that verifies if answer content is properly attributed to and supported by the provided context.",
            "user": f"""
        Context: {context}    
        Question: {question}
        Answer: {answer}

        Please look at my answer and think.
        """
        }

    def evaluate_answer(self, question: str, step_action: ExecuteAction, 
                        metrics: list[str]) -> EvaluationResponse:
        """Evaluate an answer based on given metrics"""
        for metric in metrics:
            if metric == "prompt_checker":
                checker = PromptChecker()
                return checker.evaluate_prompt(question, step_action)

            prompt = self.evluator_prompt(question, step_action.answer, step_action.context)
            result = self.llm_client.get_completion(prompt.system, prompt.user)

            evaluation_response = self.extract_evaluation_response(result)

        return evaluation_response

    def evaluate_question(self, question: str) -> list[str]:
        """Evaluate what kind of metrics should be used for this question"""
        return ["prompt_checker"]
    
    def extract_evaluation_response(self, result: str) -> EvaluationResponse:
        return json.loads(result)
