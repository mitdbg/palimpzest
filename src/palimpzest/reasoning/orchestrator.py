# agent.py
import json
from palimpzest.reasoning.action_tracker import ActionTracker
from palimpzest.reasoning.action_types import ActionType, ExecuteAction
from palimpzest.reasoning.orchestrator_prompt import orchestrator_prompt
from palimpzest.reasoning.tools.error_analyzer import ErrorAnalyzer
from palimpzest.reasoning.tools.evaluator import Evaluator
from palimpzest.reasoning.llm_api.llm_api import LLMClient
from enum import Enum


class Orchestrator:
    def __init__(self, token_budget: int = 1_000_000, max_bad_attempts: int = 3, model: str = "gpt-4o-mini"):
        self.token_budget = token_budget
        self.all_context = []
        self.max_bad_attempts = max_bad_attempts
        self.action_tracker = ActionTracker()
        self.all_knowledge = [
            "You need to take care of format transformations if users need."
        ]

        self.evaluator = Evaluator()
        self.error_analyzer = ErrorAnalyzer()
        self.master_llm_client = LLMClient(model = "gpt-4o-mini")

    def get_response(self, question: str) -> dict:
        step = 0
        total_step = 0
        bad_attempts = 0

        remaining_questions = [question]
        all_questions = [question]
        bad_context = []
        diary_context = []
        all_history = []

        this_step = None
        allow_reflect = True
        allow_answer = True
        while bad_attempts <= self.max_bad_attempts or total_step < 10:
            step += 1
            total_step += 1
            
            current_question = remaining_questions.pop(0) if remaining_questions else question
            metrics = self.evaluator.evaluate_question(current_question)
            # Generate next action
            next_action =  self._generate_next_action(
                original_question=question,
                current_question=current_question,
                bad_context=bad_context,
                diary_context=diary_context,
                all_knowledge=self.all_knowledge,
                allow_reflect=allow_reflect,
                allow_answer=allow_answer
            )

            this_step = next_action

            # Track action
            self.action_tracker.track_action({
                "total_step": total_step,
                "this_step": this_step.__dict__,
                "remaining_questions": remaining_questions,
                "bad_attempts": bad_attempts
            })

            if this_step.action == ActionType.ANSWER:
                evaluation = self.evaluator.evaluate_answer(
                    current_question, 
                    this_step,
                    metrics
                )

                if current_question == question:  # Original question
                    allow_reflect = True
                    if evaluation.pass_:
                        diary_context.append("""
At step ${step}, you took **answer** action and finally found the answer to the original question:

Original question: 
${current_question}

Your answer: 
${this_step.answer}

The evaluator thinks your answer is good because: 
${evaluation.think}

Your journey ends here. You have successfully answered the original question. Congratulations! 🎉
""")
                        this_step.is_final = True
                        break
                    else:
                        bad_attempts += 1
                        if bad_attempts >= self.max_bad_attempts:
                            this_step.is_final = False
                            break
                        diary_context.append(f"""
At step ${step}, you took **answer** action but evaluator thinks it is not a good answer:

Original question: 
${current_question}

Your answer: 
${this_step.answer}

The evaluator thinks your answer is bad because: 
${evaluation.think}
""")

                        error_analysis = self.error_analyzer.analyze_errors(
                            diary_context,
                            bad_context,
                            self.action_tracker
                        )
                        bad_context.append({
                            "question": current_question,
                            "answer": this_step.answer,
                            "evaluation": evaluation.think
                        })
                        bad_context[-1]["error_analysis"] = error_analysis
                        # Reset for next attempt
                        remaining_questions.append(question)
                        diary_context = []
                        step = 0

                elif evaluation.pass_:
                    diary_context.append(f"""
At step ${step}, you took **answer** action. You found a good answer to the sub-question:

Sub-question: 
${current_question}

Your answer: 
${this_step.answer}

The evaluator thinks your answer is good because: 
${evaluation.think}

Although you solved a sub-question, you still need to find the answer to the original question. You need to keep going.
""")
            elif this_step.action == ActionType.REFLECT:
                new_gap_questions = this_step.reflections['gaps_questions']
                allow_reflect = False
                if len(new_gap_questions) > 0:
                    new_line = "\n"
                    diary_context.append(f"""At step {step}, you took **reflect** and think about the knowledge gaps. You found some sub-questions are important to the question: "{current_question}",
                        You realize you need to know the answers to the following sub-questions: {new_line}{new_line.join([f'- {q}' for q in new_gap_questions])}
                        You will now figure out the answers to these sub-questions and see if they can help you find the answer to the original question.
                        """
                        )
                    remaining_questions.extend(new_gap_questions)
                    all_questions.extend(new_gap_questions)
                    remaining_questions.append(question)  # always keep the original question in the gaps
                else:
                    diary_context.append(
                        f"""At step {step}, you took **reflect** and think about the knowledge gaps. You tried to break down the question "{current_question}" into gap-questions like this: {new_gap_questions.join(', ')} 
                        But then you realized you have asked them before. You decided to to think out of the box or cut from a completely different angle. 
                        """
                    )
                    # diary_context.extend({
                    #     "totalStep": total_step,
                    #     "thisStep": this_step,
                    #     "result": 'You have tried all possible questions and found no useful information. You must think out of the box or different angle!!!'
                    # })

                # if "think" in this_step.reflections:
                #     new_questions = self._extract_questions(this_step.reflections['think'])
                #     remaining_questions.extend(new_questions)
                #     all_questions.extend(new_questions)
                #     remaining_questions.append(question)  # Keep original question in queue

        return {
            "result": self._make_json_serializable(this_step.__dict__),
            "context": {
                "diary": diary_context,
                "bad_context": bad_context,
                "bad_attempts": bad_attempts
            }
        }
    
    def _make_json_serializable(self, obj):
        """Convert objects to JSON serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'value') and isinstance(obj, Enum):
            return obj.value
        else:
            return obj

    def _generate_next_action(self, 
                                  original_question: str,
                                  current_question: str,
                                  diary_context: list[dict],
                                  bad_context: list[dict],
                                  all_knowledge: list[str],
                                  allow_reflect: bool,
                                  allow_answer: bool) -> dict:
        """Generate next action based on current context"""
        prompt = orchestrator_prompt(
            original_question=original_question,
            current_question=current_question,
            diary_context=diary_context,
            bad_context=bad_context,
            allow_reflect=allow_reflect,
            allow_answer=allow_answer,
            all_knowledge=all_knowledge
        )

        response = self.master_llm_client.get_completion(prompt)
        next_action = self._extract_next_action(response)
        return next_action

    def _extract_questions(self, text: str) -> list[str]:
        """Extract questions from reflection text"""
        pass

    def format_json_text(self, text: str) -> dict:
        """Format text into JSON"""
        if text.startswith("```json") and text.endswith("```"):
            # Remove markdown code block formatting
            json_text_str = text.replace("```json", "", 1).replace("```", "", 1).strip()
        else:
            # Try to find JSON content between backticks if not properly formatted
            import re
            json_match = re.search(r'```(?:json)?(.*?)```', text, re.DOTALL)
            json_text_str = json_match.group(1).strip() if json_match else text.strip()

        return json_text_str

    def _extract_next_action(self, text: str) -> dict:
        """Extract next action from text"""
        json_text = json.loads(self.format_json_text(text))
        if "action" in json_text:
            action = json_text["action"]
            if action == "action-answer":
                return ExecuteAction(
                    action=ActionType.ANSWER, 
                    answer=json_text["answer"]
                )
            elif action == "action-reflect":
                return ExecuteAction(
                    action=ActionType.REFLECT, 
                    reflections=json_text["reflections"]
                )

        return None