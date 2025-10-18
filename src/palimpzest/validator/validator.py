import json
import time

import litellm

# from colorama import Fore, Style
from palimpzest.constants import MODEL_CARDS, Cardinality, Model, PromptStrategy
from palimpzest.core.elements.records import DataRecord
from palimpzest.core.models import GenerationStats
from palimpzest.prompts import (
    FLAT_MAP_IMAGE_VALIDATOR_PROMPT,
    FLAT_MAP_VALIDATOR_PROMPT,
    MAP_IMAGE_VALIDATOR_PROMPT,
    MAP_VALIDATOR_PROMPT,
    RETRIEVE_VALIDATOR_PROMPT,
    PromptFactory,
)
from palimpzest.query.generators.generators import get_json_from_answer
from palimpzest.query.operators.convert import LLMConvert
from palimpzest.query.operators.filter import LLMFilter
from palimpzest.query.operators.join import JoinOp
from palimpzest.query.operators.topk import TopKOp


class Validator:
    """
    The Validator is used during optimization to score the output of physical operator(s) and physical plan(s).

    TODO: support end-to-end labels; will likely require a different SentinelExecutionStrategy which
          executes the full input to produce an output, evaluates the output, and then updates
          intermediate operator(s) based on the evaluation.
    """
    def __init__(self, model: Model = Model.o4_MINI):
        self.model = model
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

    def topk_score_fn(self, fields: list[str], input_record: dict, output: dict) -> float | None:
        raise NotImplementedError("Validator.map_score_fn not implemented.")

    def _get_gen_stats_from_completion(self, completion, start_time: float) -> GenerationStats:
        """
        Extract generation stats from the given completion response.
        """
        usage = completion.usage.model_dump()

        # get cost per input/output token for the model and parse number of input and output tokens
        usd_per_input_token = MODEL_CARDS[self.model.value]["usd_per_input_token"]
        usd_per_output_token = MODEL_CARDS[self.model.value]["usd_per_output_token"]
        input_tokens = usage["prompt_tokens"]
        output_tokens = usage["completion_tokens"]

        return GenerationStats(
            model_name=self.model.value,
            llm_call_duration_secs=time.time() - start_time,
            fn_call_duration_secs=0.0,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_input_cost=input_tokens * usd_per_input_token,
            total_output_cost=output_tokens * usd_per_output_token,
            cost_per_record=input_tokens * usd_per_input_token + output_tokens * usd_per_output_token,
            total_llm_calls=1,
        )

    def _default_map_score_fn(self, op: LLMConvert, fields: list[str], input_record: DataRecord, output: dict) -> tuple[float | None, GenerationStats]:
        """
        Compute the quality of the generated output for the given fields and input_record.
        """
        # create prompt factory
        factory = PromptFactory(PromptStrategy.MAP, self.model, Cardinality.ONE_TO_ONE)

        # get the input messages; strip out the system message(s)
        msg_kwargs = {"output_schema": op.output_schema, "project_cols": op.get_input_fields()}
        messages = factory.create_messages(input_record, fields, **msg_kwargs)
        input_messages = [msg for msg in messages if msg["role"] != "system"]
        output = json.dumps(output, indent=2)
        output_message = f"OUTPUT:\n--------\n{output}\n\nEVALUATION: "
        # input_str = '\n'.join(list(map(lambda d: d['content'], input_messages + [{"role": "user", "content": output_message}])))

        # invoke the judge
        score, gen_stats = None, GenerationStats()
        try:
            start_time = time.time()
            validator_prompt = MAP_IMAGE_VALIDATOR_PROMPT if op.is_image_op() else MAP_VALIDATOR_PROMPT
            val_messages = [{"role": "system", "content": validator_prompt}] + input_messages + [{"role": "user", "content": output_message}]
            completion = litellm.completion(model=self.model.value, messages=val_messages)
            completion_text = completion.choices[0].message.content
            gen_stats = self._get_gen_stats_from_completion(completion, start_time)
            # print(f"INPUT:\n{input_str}")
            # print(Fore.GREEN + f"{completion_text}\n" + Style.RESET_ALL)

            # parse the evaluation
            eval_dict: dict = get_json_from_answer(completion_text, self.model, Cardinality.ONE_TO_ONE)
            score = sum(eval_dict.values()) / len(eval_dict)

        except Exception:
            pass

        return score, gen_stats

    def _default_flat_map_score_fn(self, op: LLMConvert, fields: list[str], input_record: dict, output: list[dict]) -> tuple[float | None, GenerationStats]:
        """
        Compute the quality for each record_op_stats object in the given record_set.
        """
        # create prompt factory
        factory = PromptFactory(PromptStrategy.MAP, self.model, Cardinality.ONE_TO_MANY)

        # get the input messages; strip out the system message(s)
        msg_kwargs = {"output_schema": op.output_schema, "project_cols": op.get_input_fields()}
        messages = factory.create_messages(input_record, fields, **msg_kwargs)
        input_messages = [msg for msg in messages if msg["role"] != "system"]
        output = json.dumps(output, indent=2)
        output_message = f"OUTPUTS:\n--------\n{output}\n\nEVALUATION: "
        # input_str = '\n'.join(list(map(lambda d: d['content'], input_messages + [{"role": "user", "content": output_message}])))

        # invoke the judge
        score, gen_stats = None, GenerationStats()
        try:
            start_time = time.time()
            validator_prompt = FLAT_MAP_IMAGE_VALIDATOR_PROMPT if op.is_image_op() else FLAT_MAP_VALIDATOR_PROMPT
            val_messages = [{"role": "system", "content": validator_prompt}] + input_messages + [{"role": "user", "content": output_message}]
            completion = litellm.completion(model="openai/o4-mini", messages=val_messages)
            completion_text = completion.choices[0].message.content
            gen_stats = self._get_gen_stats_from_completion(completion, start_time)
            # print(f"INPUT:\n{input_str}")
            # print(Fore.GREEN + f"{completion_text}\n" + Style.RESET_ALL)

            # parse the evaluation
            eval_dicts: list[dict] = get_json_from_answer(completion_text, self.model, Cardinality.ONE_TO_MANY)
            all_qualities = []
            for record_eval_dict in eval_dicts:
                all_qualities.extend(record_eval_dict.values())
            score = sum(all_qualities) / len(all_qualities)

        except Exception:
            pass

        return score, gen_stats

    def _default_filter_score_fn(self, op: LLMFilter, filter_str: str, input_record: dict, output: bool) -> tuple[float | None, GenerationStats]:
        """
        Compute the quality for each record_op_stats object in the given record_set.
        """
        score, gen_stats = None, GenerationStats()
        filter_input_hash = hash(f"{filter_str}{hash(input_record)}")
        label = self.filter_cache.get(filter_input_hash, None)
        if label is None:
            validator_op: LLMFilter = op.copy()
            validator_op.model = self.model
            try:
                target_record_set = validator_op(input_record)
                label = target_record_set[0]._passed_operator
                self.filter_cache[filter_input_hash] = label
                score = float(label == output)
                record_op_stats = target_record_set.record_op_stats[0]
                gen_stats = GenerationStats(
                    model_name=self.model.value,
                    total_input_tokens=record_op_stats.total_input_tokens,
                    total_output_tokens=record_op_stats.total_output_tokens,
                    total_input_cost=record_op_stats.total_input_cost,
                    total_output_cost=record_op_stats.total_output_cost,
                    cost_per_record=record_op_stats.cost_per_record,
                    llm_call_duration_secs=record_op_stats.llm_call_duration_secs,
                    fn_call_duration_secs=record_op_stats.fn_call_duration_secs,
                    total_llm_calls=record_op_stats.total_llm_calls,
                )

            except Exception:
                pass

        else:
            score = float(label == output)

        return score, gen_stats

    def _default_join_score_fn(self, op: JoinOp, condition: str, left_input_record: DataRecord, right_input_record: DataRecord, output: bool) -> tuple[float | None, GenerationStats]:
        score, gen_stats = None, GenerationStats()
        join_input_hash = hash(f"{condition}{hash(left_input_record)}{hash(right_input_record)}")
        label = self.join_cache.get(join_input_hash, None)
        if label is None:
            validator_op: JoinOp = op.copy()
            validator_op.model = self.model
            try:
                target_record_set, _ = validator_op([left_input_record], [right_input_record])
                label = target_record_set[0]._passed_operator
                self.join_cache[join_input_hash] = label
                score = float(label == output)
                record_op_stats = target_record_set.record_op_stats[0]
                gen_stats = GenerationStats(
                    model_name=self.model.value,
                    total_input_tokens=record_op_stats.total_input_tokens,
                    total_output_tokens=record_op_stats.total_output_tokens,
                    total_input_cost=record_op_stats.total_input_cost,
                    total_output_cost=record_op_stats.total_output_cost,
                    cost_per_record=record_op_stats.cost_per_record,
                    llm_call_duration_secs=record_op_stats.llm_call_duration_secs,
                    fn_call_duration_secs=record_op_stats.fn_call_duration_secs,
                    total_llm_calls=record_op_stats.total_llm_calls,
                )

            except Exception:
                pass

        else:
            score = float(label == output)

        return score, gen_stats

    def _default_topk_score_fn(self, op: TopKOp, fields: list[str], input_record: DataRecord, output: dict) -> tuple[float | None, GenerationStats]:
        """
        Compute the quality of the generated output for the given fields and input_record.
        """
        # TODO: top-k k=25; score each item based on relevance; compute F1
        # TODO: support retrieval over images
        # create prompt factory
        factory = PromptFactory(PromptStrategy.MAP, self.model, Cardinality.ONE_TO_ONE)

        # get the input messages; strip out the system message(s)
        msg_kwargs = {"output_schema": op.output_schema, "project_cols": op.get_input_fields()}
        messages = factory.create_messages(input_record, fields, **msg_kwargs)
        input_messages = [msg for msg in messages if msg["role"] != "system"]
        output = json.dumps(output, indent=2)
        output_message = f"OUTPUT:\n--------\n{output}\n\nEVALUATION: "
        # input_str = '\n'.join(list(map(lambda d: d['content'], input_messages + [{"role": "user", "content": output_message}])))

        # invoke the judge
        score, gen_stats = None, GenerationStats()
        try:
            start_time = time.time()
            # TODO: support retrieval over images
            validator_prompt = RETRIEVE_VALIDATOR_PROMPT
            val_messages = [{"role": "system", "content": validator_prompt}] + input_messages + [{"role": "user", "content": output_message}]
            completion = litellm.completion(model="openai/o4-mini", messages=val_messages)
            completion_text = completion.choices[0].message.content
            gen_stats = self._get_gen_stats_from_completion(completion, start_time)
            # print(f"INPUT:\n{input_str}")
            # print(Fore.GREEN + f"{completion_text}\n" + Style.RESET_ALL)

            # parse the evaluation
            eval_dict: dict = get_json_from_answer(completion_text, self.model, Cardinality.ONE_TO_ONE)
            score = sum(eval_dict.values()) / len(eval_dict)

        except Exception:
            pass

        return score, gen_stats


    def _score_map(self, op: LLMConvert, fields: list[str], input_record: DataRecord, output: dict, full_hash: str) -> tuple[float | None, GenerationStats, str]:
        try:
            out = self.map_score_fn(fields, input_record.to_dict(), output)
            score, gen_stats = out if isinstance(out, tuple) else (out, GenerationStats())
            return score, gen_stats, full_hash
        except NotImplementedError:
            score, gen_stats = self._default_map_score_fn(op, fields, input_record, output)
            return score, gen_stats, full_hash

    def _score_flat_map(self, op: LLMConvert, fields: list[str], input_record: DataRecord, output: list[dict], full_hash: str) -> tuple[float | None, GenerationStats, str]:
        try:
            out = self.flat_map_score_fn(fields, input_record.to_dict(), output)
            score, gen_stats = out if isinstance(out, tuple) else (out, GenerationStats())
            return score, gen_stats, full_hash
        except NotImplementedError:
            score, gen_stats = self._default_flat_map_score_fn(op, fields, input_record, output)
            return score, gen_stats, full_hash

    def _score_filter(self, op: LLMFilter, filter_str: str, input_record: DataRecord, output: bool, full_hash: str) -> tuple[float | None, GenerationStats, str]:
        try:
            out = self.filter_score_fn(filter_str, input_record.to_dict(), output)
            score, gen_stats = out if isinstance(out, tuple) else (out, GenerationStats())
            return score, gen_stats, full_hash
        except NotImplementedError:
            score, gen_stats = self._default_filter_score_fn(op, filter_str, input_record, output)
            return score, gen_stats, full_hash

    def _score_join(self, op: JoinOp, condition: str, left_input_record: DataRecord, right_input_record: DataRecord, output: bool, full_hash: str) -> tuple[float | None, GenerationStats, str]:
        try:
            out = self.join_score_fn(condition, left_input_record.to_dict(), right_input_record.to_dict(), output)
            score, gen_stats = out if isinstance(out, tuple) else (out, GenerationStats())
            return score, gen_stats, full_hash
        except NotImplementedError:
            score, gen_stats = self._default_join_score_fn(op, condition, left_input_record, right_input_record, output)
            return score, gen_stats, full_hash

    def _score_topk(self, op: TopKOp, fields: list[str], input_record: DataRecord, output: dict, full_hash: str) -> tuple[float | None, GenerationStats, str]:
        try:
            out = self.topk_score_fn(fields, input_record.to_dict(), output)
            score, gen_stats = out if isinstance(out, tuple) else (out, GenerationStats())
            return score, gen_stats, full_hash
        except NotImplementedError:
            score, gen_stats = self._default_topk_score_fn(op, fields, input_record, output)
            return score, gen_stats, full_hash
