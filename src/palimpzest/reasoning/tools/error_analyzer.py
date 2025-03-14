import json

from palimpzest.reasoning.action_tracker import ActionTracker
from palimpzest.reasoning.llm_api.llm_api import LLMClient

def error_analyzer_prompt(diary_context: list[dict], bad_context: list[dict]) -> str:
  new_line = "\n"
  bad_context_str = ""
  diary_context_str = ""
  for d in diary_context:
    diary_context_str += json.dumps(d, indent=2) + "\n"
  for c in bad_context:
    bad_context_str += json.dumps(c, indent=2) + "\n"

  
  return {
    "system": """You are an expert at analyzing reasoning processes. Your task is to analyze the given sequence of steps and identify what went wrong in the search process.

<rules>
0. The original datasource
1. The sequence of actions taken
2. The effectiveness of each step
3. The logic between consecutive steps
4. Alternative approaches that could have been taken
5. Signs of getting stuck in repetitive patterns
6. Whether the final answer matches the accumulated information

Analyze the steps and provide detailed feedback following these guidelines:
- In the recap: Summarize key actions chronologically, highlight patterns, and identify where the process started to go wrong
- In the blame: Point to specific steps or patterns that led to the inadequate answer
- In the improvement: Provide actionable suggestions that could have led to a better outcome

Generate a JSON response following JSON schema:
 - recap: str
 - blame: str
 - improvement_plan: str
 - questions_to_answer: list[str]
</rules>

""",
    "user": f"""{diary_context_str + new_line + new_line + bad_context_str}"""
  }


class ErrorAnalyzer:
  TOOL_NAME = 'error_analyzer'

  def __init__(self, model: str = "gpt-4o-mini"):
    self.llm_client = LLMClient(model="gpt-4o-mini")

  def extract_error_analysis_response(self, result: dict) -> dict:
    if isinstance(result, str):
      try:
        if result.startswith("```json"):
          result = result.replace("```json", "", 1).replace("```", "", 1).strip()
        return json.loads(result)
      except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        # Return a default error response or the original string
        return {"error": "Failed to parse JSON response", "original": result}
    
    # If result is already a dictionary, return it as is
    return result

  def analyze_errors(self, diary_context: list[dict], bad_context: list[dict], trackers: ActionTracker) -> dict:
    try:
      prompt = error_analyzer_prompt(diary_context, [])

      result = self.llm_client.get_completion(
        prompt="",
        user_prompt=prompt["user"],
        system_prompt=prompt["system"]
      )

      error_analysis_response = self.extract_error_analysis_response(result)
      return error_analysis_response

    except Exception as error:
      print(f"Error in {self.TOOL_NAME}", error)
      raise error




