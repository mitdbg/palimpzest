# from abc import ABC, abstractmethod
import json
from typing import Callable

import litellm

# from colorama import Fore, Style
from palimpzest.constants import Cardinality, Model, PromptStrategy
from palimpzest.core.elements.records import DataRecordSet
from palimpzest.prompts import VALIDATOR_PROMPT, PromptFactory
from palimpzest.query.generators.generators import get_json_from_answer
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.physical import PhysicalOperator


class BaseValidator:
    """
    The Validator is used during optimization to score the output of physical operator(s) and physical plan(s).

    The core function of the Validator is to take a (set of) input(s) and a (set of) output(s)
    - LLM validation vs. Non-LLM validation
    - operator-level validation vs. plan-level validation
    - LLM validation may only make sense at the operator-level
    - Non-LLM Validation may work at the operator and/or plan-level

    TODO: start with non-llm based operator level validation; port over code from Sentinel executor
    TODO: allow Validator to come with its own source Dataset(s)
    TODO: try to eliminate need for source_idx
    """
    def __init__(self, eval_fn: Callable | None = None) -> None:
        self.eval_fn = self.default_eval_fn if eval_fn is None else eval_fn

    def default_eval_fn(self, record_set_tuples: list[tuple[DataRecordSet, PhysicalOperator]]) -> list[tuple[DataRecordSet, PhysicalOperator]]:
        """
        Compute the quality for each record_op_stats object in the given record_set.
        """
        # TODO: COT_BOOL_IMAGE; COT_QA_IMAGE; Cardinality and Model
        label = None
        for record_set, physical_op in record_set_tuples:
            is_filter_op = isinstance(physical_op, LLMFilter)

            # isolate record_op_stats
            record_op_stats = record_set.record_op_stats

            # create prompt factory
            prompt_strategy = PromptStrategy.COT_BOOL if is_filter_op else PromptStrategy.COT_QA
            factory = PromptFactory(prompt_strategy, Model.GPT_4o, Cardinality.ONE_TO_ONE)

            # get the input messages; strip out the system message(s)
            output_fields = ["passed_operator"] if is_filter_op else record_set.record_op_stats[0].generated_fields
            msg_kwargs = {"filter_condition": record_op_stats[0].filter_str} if is_filter_op else {"output_schema": record_set.schema}
            messages = factory.create_messages(record_set.input_data_record, output_fields, **msg_kwargs)
            input_messages = [msg for msg in messages if msg["role"] != "system"]

            outputs = []
            for record_op_stats in record_set.record_op_stats:
                output_dict = (
                    {"passed_operator": record_op_stats.passed_operator}
                    if record_op_stats.generated_fields is None
                    else {k: v for k, v in record_op_stats.record_state.items() if k in record_op_stats.generated_fields}
                )
                output_dict["record_id"] = record_op_stats.record_id
                outputs.append(output_dict)

            outputs = json.dumps(outputs, indent=2)
            outputs_message = f"OUTPUTS:\n--------\n{outputs}\n\nEVALUATION: "
            # input_str = '\n'.join(list(map(lambda d: d['content'], input_messages + [{"role": "user", "content": outputs_message}])))

            # if this operation failed
            if len(record_set) == 0:
                record_set.record_op_stats[0].quality = 0.0
                continue

            # if this is a filter op, run the op with the validator and assess quality relative to its output
            if is_filter_op and label is None:
                validator_op = physical_op.copy()
                validator_op.model = Model.GPT_4o
                try:
                    target_record_set = validator_op(record_set.input_data_record)
                    label = target_record_set.record_op_stats[0].passed_operator
                    # print(f"INPUT:\n{input_str}")
                    # print(Fore.GREEN + f"VALIDATOR LABEL: {label}\n" + Style.RESET_ALL)
                except Exception:
                    label = None

            # apply label for filters
            if is_filter_op:
                record_set.record_op_stats[0].quality = (
                    (label == record_set.record_op_stats[0].passed_operator)
                    if label is not None
                    else 0.5
                )

            else:
                # invoke the judge
                try:
                    val_messages = [{"role": "system", "content": VALIDATOR_PROMPT}] + input_messages + [{"role": "user", "content": outputs_message}]
                    completion = litellm.completion(model="openai/gpt-4o", messages=val_messages)
                    completion_text = completion.choices[0].message.content
                    # print(f"INPUT:\n{input_str}")
                    # print(Fore.GREEN + f"{completion_text}\n" + Style.RESET_ALL)

                    # parse the evaluation
                    output: list[dict] = get_json_from_answer(completion_text, Model.GPT_4o, Cardinality.ONE_TO_MANY)
                    for record_eval_dict in output:
                        for record_op_stats in record_set.record_op_stats:
                            if record_eval_dict["record_id"] == record_op_stats.record_id:
                                record_eval_dict.pop("record_id")
                                record_op_stats.quality = sum(record_eval_dict.values()) / len(record_eval_dict)
                except Exception:
                    for record_op_stats in record_set.record_op_stats:
                        record_op_stats.quality = 0.5

        return record_set


class Validator(BaseValidator):
    """
    """
    pass


class LLMValidator(BaseValidator):
    """
    """
    pass
