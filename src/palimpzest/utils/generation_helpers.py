from typing import Any, Dict

import json
import re


def getJsonFromAnswer(answer: str) -> Dict[str, Any]:
    """
    This function parses an LLM response which is supposed to output a JSON object
    and optimistically searches for the substring containing the JSON object.
    """
    if not answer.strip().startswith("{"):
        # Find the start index of the actual JSON string
        # assuming the prefix is followed by the JSON object/array
        start_index = answer.find("{") if "{" in answer else answer.find("[")
        if start_index != -1:
            # Remove the prefix and any leading characters before the JSON starts
            answer = answer[start_index:]

    if not answer.strip().endswith("}"):
        # Find the end index of the actual JSON string
        # assuming the suffix is preceded by the JSON object/array
        end_index = answer.rfind("}") if "}" in answer else answer.rfind("]")
        if end_index != -1:
            # Remove the suffix and any trailing characters after the JSON ends
            answer = answer[: end_index + 1]

    # Handle weird escaped values. I am not sure why the model
    # is returning these, but the JSON parser can't take them
    answer = answer.replace("\_", "_")

    # Handle comments in the JSON response. Use regex from // until end of line
    answer = re.sub(r"\/\/.*$", "", answer, flags=re.MULTILINE)
    return json.loads(answer)
