import json

import litellm

from colorama import Fore, Style
from palimpzest.constants import Cardinality, Model, PromptStrategy
from palimpzest.core.elements.records import DataRecord
from palimpzest.prompts import (
    FLAT_MAP_IMAGE_VALIDATOR_PROMPT,
    FLAT_MAP_VALIDATOR_PROMPT,
    MAP_IMAGE_VALIDATOR_PROMPT,
    MAP_VALIDATOR_PROMPT,
    PromptFactory,
)
from palimpzest.query.generators.generators import get_json_from_answer
from palimpzest.query.operators.convert import LLMConvert
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.join import JoinOp


class Validator:
    """
    The Validator is used during optimization to score the output of physical operator(s) and physical plan(s).

    TODO: support end-to-end labels; will likely require a different SentinelExecutionStrategy which
          executes the full input to produce an output, evaluates the output, and then updates
          intermediate operator(s) based on the evaluation.
    """
    def __init__(self):
        self.filter_cache = {}
        self.join_cache = {}

    def map_score_fn(self, fields: list[str], input_record: dict, output: dict) -> float | None:
        raise NotImplementedError("Validator.map_score_fn not implemented.")

    def flat_map_score_fn(self, fields: list[str], input_record: dict, output: list[dict]) -> float | None:
        raise NotImplementedError("Validator.flat_map_score_fn not implemented.")

    def filter_score_fn(self, filter_str: str, input_record: dict, output: bool) -> float | None:
        raise NotImplementedError("Validator.filter_score_fn not implemented.")

    def join_score_fn(self, condition: str, left_input_record: dict, right_input_record: dict, output: bool) -> float | None:
        raise NotImplementedError("Validator.join_score_fn not implemented.")

    # TODO: cache map outputs and their scores, as common field extractions are likely to repeat
    def _default_map_score_fn(self, op: LLMConvert, fields: list[str], input_record: DataRecord, output: dict) -> float | None:
        """
        Compute the quality of the generated output for the given fields and input_record.
        """
        # create prompt factory
        prompt_strategy = PromptStrategy.COT_QA_IMAGE if op.is_image_conversion() else PromptStrategy.COT_QA
        factory = PromptFactory(prompt_strategy, Model.GPT_4o, Cardinality.ONE_TO_ONE) # TODO: switch to o4_MINI after merging in dev

        # get the input messages; strip out the system message(s)
        msg_kwargs = {"output_schema": op.output_schema, "project_cols": op.get_input_fields()}
        messages = factory.create_messages(input_record, fields, **msg_kwargs)
        input_messages = [msg for msg in messages if msg["role"] != "system"]
        output = json.dumps(output, indent=2)
        output_message = f"OUTPUT:\n--------\n{output}\n\nEVALUATION: "
        input_str = '\n'.join(list(map(lambda d: d['content'], input_messages + [{"role": "user", "content": output_message}])))

        # invoke the judge
        score = None
        try:
            validator_prompt = MAP_IMAGE_VALIDATOR_PROMPT if op.is_image_conversion() else MAP_VALIDATOR_PROMPT
            val_messages = [{"role": "system", "content": validator_prompt}] + input_messages + [{"role": "user", "content": output_message}]
            completion = litellm.completion(model="openai/o4-mini", messages=val_messages)
            completion_text = completion.choices[0].message.content
            print(f"INPUT:\n{input_str}")
            print(Fore.GREEN + f"{completion_text}\n" + Style.RESET_ALL)

            # parse the evaluation
            eval_dict: dict = get_json_from_answer(completion_text, Model.GPT_4o, Cardinality.ONE_TO_ONE) # TODO: modify VALIDATOR_PROMPT above to expect single dict output
            score = sum(eval_dict.values()) / len(eval_dict)

        except Exception:
            pass

        return score

    def _default_flat_map_score_fn(self, op: LLMConvert, fields: list[str], input_record: dict, output: list[dict]) -> float | None:
        """
        Compute the quality for each record_op_stats object in the given record_set.
        """
        # create prompt factory
        prompt_strategy = PromptStrategy.COT_QA_IMAGE if op.is_image_conversion() else PromptStrategy.COT_QA
        factory = PromptFactory(prompt_strategy, Model.GPT_4o, Cardinality.ONE_TO_MANY) # TODO: switch to o4_MINI after merging in dev

        # get the input messages; strip out the system message(s)
        msg_kwargs = {"output_schema": op.output_schema, "project_cols": op.get_input_fields()}
        messages = factory.create_messages(input_record, fields, **msg_kwargs)
        input_messages = [msg for msg in messages if msg["role"] != "system"]
        output = json.dumps(output, indent=2)
        output_message = f"OUTPUTS:\n--------\n{output}\n\nEVALUATION: "
        # input_str = '\n'.join(list(map(lambda d: d['content'], input_messages + [{"role": "user", "content": output_message}])))

        # invoke the judge
        score = None
        try:
            validator_prompt = FLAT_MAP_IMAGE_VALIDATOR_PROMPT if op.is_image_conversion() else FLAT_MAP_VALIDATOR_PROMPT
            val_messages = [{"role": "system", "content": validator_prompt}] + input_messages + [{"role": "user", "content": output_message}]
            completion = litellm.completion(model="openai/o4-mini", messages=val_messages)
            completion_text = completion.choices[0].message.content
            # print(f"INPUT:\n{input_str}")
            # print(Fore.GREEN + f"{completion_text}\n" + Style.RESET_ALL)

            # parse the evaluation
            eval_dicts: list[dict] = get_json_from_answer(completion_text, Model.GPT_4o, Cardinality.ONE_TO_MANY)
            all_qualities = []
            for record_eval_dict in eval_dicts:
                all_qualities.extend(record_eval_dict.values())
            score = sum(all_qualities) / len(all_qualities)

        except Exception:
            pass

        return score

    def _default_filter_score_fn(self, op: LLMFilter, filter_str: str, input_record: dict, output: bool) -> float | None:
        """
        Compute the quality for each record_op_stats object in the given record_set.
        """
        score = None
        filter_input_hash = hash(f"{filter_str}{hash(input_record)}")
        label = self.filter_cache.get(filter_input_hash, None)
        if label is None:
            validator_op: LLMFilter = op.copy()
            validator_op.model = Model.GPT_4o
            try:
                target_record_set = validator_op(input_record)
                label = target_record_set[0].passed_operator
                self.filter_cache[filter_input_hash] = label
                score = label == output

            except Exception:
                pass

        else:
            score = label == output

        return score

    def _default_join_score_fn(self, op: JoinOp, condition: str, left_input_record: DataRecord, right_input_record: DataRecord, output: bool) -> float | None:
        score = None
        join_input_hash = hash(f"{condition}{hash(left_input_record)}{hash(right_input_record)}")
        label = self.join_cache.get(join_input_hash, None)
        if label is None:
            validator_op: JoinOp = op.copy()
            validator_op.model = Model.GPT_4o
            try:
                target_record_set = validator_op([left_input_record], [right_input_record])
                label = target_record_set[0].passed_operator
                self.join_cache[join_input_hash] = label
                score = label == output

            except Exception:
                pass

        else:
            score = label == output

        return score


    def _score_map(self, op: LLMConvert, fields: list[str], input_record: DataRecord, output: dict):
        try:
            return self.map_score_fn(fields, input_record.to_dict(), output)
        except NotImplementedError:
            return self._default_map_score_fn(op, fields, input_record, output)

    def _score_flat_map(self, op: LLMConvert, fields: list[str], input_record: DataRecord, output: list[dict]):
        try:
            return self.flat_map_score_fn(fields, input_record.to_dict(), output)
        except NotImplementedError:
            return self._default_flat_map_score_fn(op, fields, input_record, output)

    def _score_filter(self, op: LLMFilter, filter_str: str, input_record: DataRecord, output: bool):
        try:
            return self.filter_score_fn(filter_str, input_record.to_dict(), output)
        except NotImplementedError:
            return self._default_filter_score_fn(op, filter_str, input_record, output)

    def _score_join(self, op: JoinOp, condition: str, left_input_record: DataRecord, right_input_record: DataRecord, output: bool):
        try:
            return self.join_score_fn(condition, left_input_record.to_dict(), right_input_record.to_dict(), output)
        except NotImplementedError:
            return self._default_join_score_fn(op, condition, left_input_record, right_input_record, output)
