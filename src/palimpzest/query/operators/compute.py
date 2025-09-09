import functools
import inspect
import os
import time
from typing import Any

from smolagents import CodeAgent, LiteLLMModel, tool

from palimpzest.core.data.context import Context
from palimpzest.core.data.context_manager import ContextManager
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import GenerationStats, OperatorCostEstimates, RecordOpStats
from palimpzest.query.operators.physical import PhysicalOperator

# TODO: need to store final executed code in compute() operator so that humans can debug when human-in-the-loop

def make_tool(bound_method):
    # Get the original function and bound instance
    func = bound_method.__func__
    instance = bound_method.__self__
    
    # Get the signature and remove 'self'
    sig = inspect.signature(func)
    params = list(sig.parameters.values())[1:]  # skip 'self'
    new_sig = inspect.Signature(parameters=params, return_annotation=sig.return_annotation)

    # Create a wrapper function dynamically
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(instance, *args, **kwargs)

    # Update the __signature__ to reflect the new one without 'self'
    wrapper.__signature__ = new_sig

    return wrapper


class SmolAgentsCompute(PhysicalOperator):
    """
    """
    def __init__(self, context_id: str, instruction: str, additional_contexts: list[Context] | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_id = context_id
        self.instruction = instruction
        self.additional_contexts = [] if additional_contexts is None else additional_contexts
        # self.model_id = "anthropic/claude-3-7-sonnet-latest"
        self.model_id = "openai/gpt-4o-mini-2024-07-18"
        # self.model_id = "openai/gpt-4o-2024-08-06"
        api_key = os.getenv("ANTHROPIC_API_KEY") if "anthropic" in self.model_id else os.getenv("OPENAI_API_KEY")
        self.model = LiteLLMModel(model_id=self.model_id, api_key=api_key)

    def __str__(self):
        op = super().__str__()
        op += f"    Context ID: {self.context_id:20s}\n"
        op += f"    Instruction: {self.instruction:20s}\n"
        op += f"    Add. Ctxs: {self.additional_contexts}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {
            "context_id": self.context_id,
            "instruction": self.instruction,
            "additional_contexts": self.additional_contexts,
            **id_params,
        }

    def get_op_params(self):
        op_params = super().get_op_params()
        return {
            "context_id": self.context_id,
            "instruction": self.instruction,
            "additional_contexts": self.additional_contexts,
            **op_params,
        }

    def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
        return OperatorCostEstimates(
            cardinality=source_op_cost_estimates.cardinality,
            time_per_record=100,
            cost_per_record=1,
            quality=1.0,
        )

    def _create_record_set(
        self,
        candidate: DataRecord,
        generation_stats: GenerationStats,
        total_time: float,
        answer: dict[str, Any],
    ) -> DataRecordSet:
        """
        Given an input DataRecord and a determination of whether it passed the filter or not,
        construct the resulting RecordSet.
        """
        # create new DataRecord
        data_item = {field: answer[field] for field in self.output_schema.model_fields if field in answer}
        dr = DataRecord.from_parent(self.output_schema, data_item, parent_record=candidate)

        # create RecordOpStats object
        record_op_stats = RecordOpStats(
            record_id=dr._id,
            record_parent_ids=dr._parent_ids,
            record_source_indices=dr._source_indices,
            record_state=dr.to_dict(include_bytes=False),
            full_op_id=self.get_full_op_id(),
            logical_op_id=self.logical_op_id,
            op_name=self.op_name(),
            time_per_record=total_time,
            cost_per_record=generation_stats.cost_per_record,
            model_name=self.get_model_name(),
            total_input_tokens=generation_stats.total_input_tokens,
            total_output_tokens=generation_stats.total_output_tokens,
            total_input_cost=generation_stats.total_input_cost,
            total_output_cost=generation_stats.total_output_cost,
            llm_call_duration_secs=generation_stats.llm_call_duration_secs,
            fn_call_duration_secs=generation_stats.fn_call_duration_secs,
            total_llm_calls=generation_stats.total_llm_calls,
            total_embedding_llm_calls=generation_stats.total_embedding_llm_calls,
            answer={k: v.description if isinstance(v, Context) else v for k, v in answer.items()},
            op_details={k: str(v) for k, v in self.get_id_params().items()},
        )

        return DataRecordSet([dr], [record_op_stats])

    def __call__(self, candidate: DataRecord) -> Any:
        start_time = time.time()

        # get the input context object and its tools
        input_context: Context = candidate.context
        description = input_context.description
        tools = [tool(make_tool(f)) for f in input_context.tools]

        # update the description to include any additional contexts
        for ctx in self.additional_contexts:
            # TODO: remove additional context if it is an ancestor of the input context
            # (not just if it is equal to the input context)
            if ctx.id == input_context.id:
                continue
            description += f"\n\nHere is some additional Context which may be useful:\n\n{ctx.description}"

        # perform the computation
        instructions = f"\n\nHere is a description of the Context whose data you will be working with, as well as any previously computed results:\n\n{description}"
        agent = CodeAgent(
            tools=tools,
            model=self.model,
            add_base_tools=False,
            instructions=instructions,
            return_full_result=True,
            additional_authorized_imports=["pandas", "io", "os"],
            planning_interval=4,
            max_steps=30,
        )
        result = agent.run(self.instruction)
        # NOTE: you can see the system prompt with `agent.memory.system_prompt.system_prompt`
        # full_steps = agent.memory.get_full_steps()

        # compute generation stats
        response = result.output
        input_tokens = result.token_usage.input_tokens
        output_tokens = result.token_usage.output_tokens
        cost_per_input_token = (3.0 / 1e6) if "anthropic" in self.model_id else (0.15 / 1e6) # (2.5 / 1e6) #
        cost_per_output_token = (15.0 / 1e6) if "anthropic" in self.model_id else (0.6 / 1e6) # (10.0 / 1e6) #
        input_cost = input_tokens * cost_per_input_token
        output_cost = output_tokens * cost_per_output_token
        generation_stats = GenerationStats(
            model_name=self.model_id,
            total_input_tokens=input_tokens,
            total_output_tokens=output_tokens,
            total_input_cost=input_cost,
            total_output_cost=output_cost,
            cost_per_record=input_cost + output_cost,
            llm_call_duration_secs=time.time() - start_time,
        )

        # update the description of the computed Context to include the result
        new_description = f"RESULT: {response}\n\n"
        cm = ContextManager()
        cm.update_context(id=self.context_id, description=new_description)

        # create and return record set
        field_answers = {
            "context": cm.get_context(id=self.context_id),
            f"result-{self.context_id}": response,
        }
        record_set = self._create_record_set(
            candidate,
            generation_stats,
            time.time() - start_time,
            field_answers,
        )

        return record_set

# import json; json.dumps(agent.memory.get_full_steps())
# agent.memory.get_full_steps()[1].keys()
# dict_keys(['step_number', 'timing', 'model_input_messages', 'tool_calls', 'error', 'model_output_message', 'model_output', 'code_action', 'observations', 'observations_images', 
# 'action_output', 'token_usage', 'is_final_answer'])
# agent.memory.get_full_steps()[1]['action_output']