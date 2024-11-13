import json
from typing import Any, Dict

import regex as re  # Use regex instead of re to used variable length lookbehind


def get_json_from_answer(answer: str) -> Dict[str, Any]:
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
    answer = answer.replace(r"\_", "_")
    answer = answer.replace("\\n", "\n")
    # Remove https and http prefixes to not conflict with comment detection
    # Handle comments in the JSON response. Use regex from // until end of line
    answer = re.sub(r"(?<!https?:)\/\/.*?$", "", answer, flags=re.MULTILINE)
    answer = re.sub(r",\n.*\.\.\.$", "", answer, flags=re.MULTILINE)
    # Sanitize newlines in the JSON response
    answer = answer.replace("\n", " ")
    try:
        response = json.loads(answer)
    except Exception as e:
        if "items" in answer:  # If we are in one to many
            # Find the last dictionary item not closed
            last_idx = answer.rfind("},")
            # Close the last dictionary item
            answer = answer[: last_idx + 1] + "]}"
            response = json.loads(answer)
        else:
            raise e
    return response
