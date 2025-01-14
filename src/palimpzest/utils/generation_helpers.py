import json
from typing import Any

import regex as re  # Use regex instead of re to used variable length lookbehind

from palimpzest.constants import Cardinality, Model


def get_json_from_answer(answer: str, model: Model, cardinality: Cardinality) -> dict[str, Any]:
    """
    This function parses an LLM response which is supposed to output a JSON object
    and optimistically searches for the substring containing the JSON object.
    """
    # model-specific trimming for LLAMA3 responses
    if model in [Model.LLAMA3, Model.LLAMA3_V]:
        answer = answer.split("---")[0]
        answer = answer.replace("True", "true")
        answer = answer.replace("False", "false")

    # split off context / excess, which models sometimes output after answer
    answer = answer.split("Context:")[0]
    answer = answer.split("# this is the answer")[0]

    # trim the answer to only include the JSON dictionary
    if cardinality == Cardinality.ONE_TO_ONE:
        if not answer.strip().startswith("{"):
            # Find the start index of the actual JSON string assuming the prefix is followed by the JSON dictionary
            start_index = answer.find("{")
            if start_index != -1:
                # Remove the prefix and any leading characters before the JSON starts
                answer = answer[start_index:]

        if not answer.strip().endswith("}"):
            # Find the end index of the actual JSON string assuming the suffix is preceded by the JSON dictionary
            end_index = answer.rfind("}")
            if end_index != -1:
                # Remove the suffix and any trailing characters after the JSON ends
                answer = answer[: end_index + 1]

    # otherwise, trim the answer to only include the JSON array
    else:
        if not answer.strip().startswith("["):
            # Find the start index of the actual JSON string assuming the prefix is followed by the JSON array
            start_index = answer.find("[")
            if start_index != -1:
                # Remove the prefix and any leading characters before the JSON starts
                answer = answer[start_index:]

        if not answer.strip().endswith("]"):
            # Find the end index of the actual JSON string
            # assuming the suffix is preceded by the JSON object/array
            end_index = answer.rfind("]")
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

    # finally, parse and return the JSON object; errors are handled by the caller
    return json.loads(answer)
