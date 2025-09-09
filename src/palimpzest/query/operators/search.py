import functools
import inspect
import os
import time
from typing import Any

# from mem0 import Memory
from smolagents import CodeAgent, LiteLLMModel, tool

# from palimpzest.agents.search_agents import DataDiscoveryAgent, SearchManagerAgent
from palimpzest.core.data.context import Context
from palimpzest.core.data.context_manager import ContextManager
from palimpzest.core.elements.records import DataRecord, DataRecordSet
from palimpzest.core.models import GenerationStats, OperatorCostEstimates, RecordOpStats
from palimpzest.query.operators.physical import PhysicalOperator


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


class SmolAgentsSearch(PhysicalOperator):
    """
    Physical operator for searching with Smol Agents.
    """
    def __init__(self, context_id: str, search_query: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_id = context_id
        self.search_query = search_query
        # self.model_id = "anthropic/claude-3-7-sonnet-latest"
        self.model_id = "openai/gpt-4o-mini-2024-07-18"
        # self.model_id = "openai/gpt-4o-2024-08-06"
        api_key = os.getenv("ANTHROPIC_API_KEY") if "anthropic" in self.model_id else os.getenv("OPENAI_API_KEY")
        self.model = LiteLLMModel(model_id=self.model_id, api_key=api_key)

    def __str__(self):
        op = super().__str__()
        op += f"    Context ID: {self.context_id:20s}\n"
        op += f"    Search Query: {self.search_query:20s}\n"
        return op

    def get_id_params(self):
        id_params = super().get_id_params()
        return {
            "context_id": self.context_id,
            "search_query": self.search_query,
            **id_params,
        }

    def get_op_params(self):
        op_params = super().get_op_params()
        return {
            "context_id": self.context_id,
            "search_query": self.search_query,
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

        # # construct the full search query
        # full_query = f"Please execute the following search query. Output a **detailed** description of (1) which data you look at, and (2) what you find in that data. Avoid making overly broad statements such as \"What you're searching for is not present in the dataset\". Instead, make more precise statments like \"What you're searching for is not present in files A.txt, B.txt, and C.txt, but may be present elsewhere\".\n\nQUERY: {self.search_query}"

        # perform the computation
        instructions = f"\n\nHere is a description of the Context whose data you will be working with, as well as any previously computed results:\n\n{description}"
        agent = CodeAgent(
            tools=tools,
            model=self.model,
            add_base_tools=False,
            instructions=instructions,
            return_full_result=True,
            additional_authorized_imports=["pandas", "io", "os"],
        )
        result = agent.run(self.search_query)
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

        # update the description of the Context to include the search result
        new_description = f"RESULT: {response}\n\n"
        cm = ContextManager()
        cm.update_context(id=self.context_id, description=new_description)

        # create and return record set
        field_answers = {
            "context": cm.get_context(id=self.context_id),
        }
        record_set = self._create_record_set(
            candidate,
            generation_stats,
            time.time() - start_time,
            field_answers,
        )

        return record_set


# class SmolAgentsManagedSearch(PhysicalOperator):
#     """
#     Physical operator for searching with Smol Agents using an Orchestrator and a Data Discovery Agent.
#     """
#     def __init__(self, context_id: str, search_query: str, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.context_id = context_id
#         self.search_query = search_query
#         # self.model_id = "anthropic/claude-3-7-sonnet-latest"
#         self.model_id = "openai/gpt-4o-mini-2024-07-18"
#         # self.model_id = "o1"
#         model_params = {
#             "model_id": self.model_id,
#             "custom_role_conversions": {"tool-call": "assistant", "tool-response": "user"},
#             "max_completion_tokens": 8192,
#         }
#         if self.model_id == "o1":
#             model_params["reasoning_effort"] = "high"
#         self.model = LiteLLMModel(**model_params)
#         self.text_limit = 100000
#         self.memory = Memory()

#     def __str__(self):
#         op = super().__str__()
#         op += f"    Context ID: {self.context_id:20s}\n"
#         op += f"    Search Query: {self.search_query:20s}\n"
#         return op

#     def get_id_params(self):
#         id_params = super().get_id_params()
#         return {
#             "context_id": self.context_id,
#             "search_query": self.search_query,
#             **id_params,
#         }

#     def get_op_params(self):
#         op_params = super().get_op_params()
#         return {
#             "context_id": self.context_id,
#             "search_query": self.search_query,
#             **op_params,
#         }

#     def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
#         return OperatorCostEstimates(
#             cardinality=source_op_cost_estimates.cardinality,
#             time_per_record=100,
#             cost_per_record=1,
#             quality=1.0,
#         )

#     def _create_record_set(
#         self,
#         candidate: DataRecord,
#         generation_stats: GenerationStats,
#         total_time: float,
#         answer: dict[str, Any],
#     ) -> DataRecordSet:
#         """
#         Given an input DataRecord and a determination of whether it passed the filter or not,
#         construct the resulting RecordSet.
#         """
#         # create new DataRecord
#         data_item = {field: answer[field] for field in self.output_schema.model_fields if field in answer}
#         dr = DataRecord.from_parent(self.output_schema, data_item, parent_record=candidate)

        # # create RecordOpStats object
        # record_op_stats = RecordOpStats(
        #     record_id=dr._id,
        #     record_parent_ids=dr._parent_ids,
        #     record_source_indices=dr._source_indices,
        #     record_state=dr.to_dict(include_bytes=False),
        #     full_op_id=self.get_full_op_id(),
        #     logical_op_id=self.logical_op_id,
        #     op_name=self.op_name(),
        #     time_per_record=total_time,
        #     cost_per_record=generation_stats.cost_per_record,
        #     model_name=self.get_model_name(),
        #     total_input_tokens=generation_stats.total_input_tokens,
        #     total_output_tokens=generation_stats.total_output_tokens,
        #     total_input_cost=generation_stats.total_input_cost,
        #     total_output_cost=generation_stats.total_output_cost,
        #     llm_call_duration_secs=generation_stats.llm_call_duration_secs,
        #     fn_call_duration_secs=generation_stats.fn_call_duration_secs,
        #     total_llm_calls=generation_stats.total_llm_calls,
        #     total_embedding_llm_calls=generation_stats.total_embedding_llm_calls,
        #     answer={k: v.description if isinstance(v, Context) else v for k, v in answer.items()},
        #     op_details={k: str(v) for k, v in self.get_id_params().items()},
        # )

#         return DataRecordSet([dr], [record_op_stats])

#     def __call__(self, candidate: DataRecord) -> Any:
#         start_time = time.time()

#         # get the input context object and its tools
#         input_context: Context = candidate.context
#         description = input_context.description
#         tools = [tool(make_tool(f)) for f in input_context.tools]

#         # create a memory tool for accessing past searches
#         @tool
#         def tool_search_history(query: str) -> str:
#             """
#             This tool enables the agent to search through its history of execution in previous sessions.
#             Thus, the agent can learn more about what it has done in the past by invoking this tool with
#             a query describing what past interactions the agent might be curious about.

#             Args:
#                 query (str): A description of what the agent wishes to search for in its execution history.

#             Returns:
#                 str: A summary of the agent execution history which is relevant to the query.
#             """
#             memories = self.memory.search(query=query, user_id="data_discovery_agent")
#             memory_str = ""
#             for idx, memory in enumerate(memories):
#                 memory_str += f"MEMORY {idx+1}: {memory['memory']}"
#             return memory_str

#         # tools.append(tool_search_history)
#         data_discovery_agent = CodeAgent(
#             model=self.model,
#             tools=tools,
#             max_steps=20,
#             verbosity_level=2,
#             planning_interval=4,
#             name="data_discovery_agent",
#             description="""A team member that will search a data repository to find files which help to answer your question.
#         Ask him for all your questions that require searching a repository of relevant data.
#         Provide him as much context as possible, in particular if you need to search on a specific timeframe!
#         And don't hesitate to provide him with a complex search task, like finding a difference between two files.
#         Your request must be a real sentence, not a keyword search! Like "Find me this information (...)" rather than a few keywords.
#         """,
#             provide_run_summary=True,
#         )
#         data_discovery_agent.prompt_templates["managed_agent"]["task"] += f"""\n\nHere is a description of the context you will be working with: {description}\n\nSearch as many files as possible before returning your final answer.\n\nAdditionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""

#         manager_agent = CodeAgent(
#             model=self.model,
#             tools=tools,
#             max_steps=12,
#             verbosity_level=2,
#             additional_authorized_imports=["*"],
#             planning_interval=4,
#             managed_agents=[data_discovery_agent],
#             return_full_result=True,
#         )

#         # TODO: improve context descriptions and add memory from there; expand to multi-modal benchmark(s)
#         # perform the computation
#         result = manager_agent.run(self.search_query)

#         # compute generation stats
#         response = result.output
#         input_tokens = result.token_usage.input_tokens
#         output_tokens = result.token_usage.output_tokens
#         cost_per_input_token = (3.0 / 1e6) if "anthropic" in self.model_id else (0.15 / 1e6) # (15.0 / 1e6)
#         cost_per_output_token = (15.0 / 1e6) if "anthropic" in self.model_id else (0.6 / 1e6) # (60.0 / 1e6)
#         input_cost = input_tokens * cost_per_input_token
#         output_cost = output_tokens * cost_per_output_token
#         generation_stats = GenerationStats(
#             model_name=self.model_id,
#             total_input_tokens=input_tokens,
#             total_output_tokens=output_tokens,
#             total_input_cost=input_cost,
#             total_output_cost=output_cost,
#             cost_per_record=input_cost + output_cost,
#             llm_call_duration_secs=time.time() - start_time,
#         )

#         # update the description of the Context to include the search result
#         new_description = f"RESULT: {response}\n\n"
#         cm = ContextManager()
#         cm.update_context(id=self.context_id, description=new_description)

#         # create and return record set
#         field_answers = {
#             "context": cm.get_context(id=self.context_id),
#         }
#         record_set = self._create_record_set(
#             candidate,
#             generation_stats,
#             time.time() - start_time,
#             field_answers,
#         )

#         return record_set


# class SmolAgentsCustomManagedSearch(PhysicalOperator):
#     """
#     Physical operator for searching with Smol Agents using an Orchestrator and a Data Discovery Agent.
#     """
#     def __init__(self, context_id: str, search_query: str, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.context_id = context_id
#         self.search_query = search_query
#         # self.model_id = "anthropic/claude-3-7-sonnet-latest"
#         self.model_id = "openai/gpt-4o-mini-2024-07-18"
#         # self.model_id = "o1"
#         model_params = {
#             "model_id": self.model_id,
#             "custom_role_conversions": {"tool-call": "assistant", "tool-response": "user"},
#             "max_completion_tokens": 8192,
#         }
#         if self.model_id == "o1":
#             model_params["reasoning_effort"] = "high"
#         self.model = LiteLLMModel(**model_params)
#         self.text_limit = 100000

#     def __str__(self):
#         op = super().__str__()
#         op += f"    Context ID: {self.context_id:20s}\n"
#         op += f"    Search Query: {self.search_query:20s}\n"
#         return op

#     def get_id_params(self):
#         id_params = super().get_id_params()
#         return {
#             "context_id": self.context_id,
#             "search_query": self.search_query,
#             **id_params,
#         }

#     def get_op_params(self):
#         op_params = super().get_op_params()
#         return {
#             "context_id": self.context_id,
#             "search_query": self.search_query,
#             **op_params,
#         }

#     def naive_cost_estimates(self, source_op_cost_estimates: OperatorCostEstimates) -> OperatorCostEstimates:
#         return OperatorCostEstimates(
#             cardinality=source_op_cost_estimates.cardinality,
#             time_per_record=100,
#             cost_per_record=1,
#             quality=1.0,
#         )

#     def _create_record_set(
#         self,
#         candidate: DataRecord,
#         generation_stats: GenerationStats,
#         total_time: float,
#         answer: dict[str, Any],
#     ) -> DataRecordSet:
#         """
#         Given an input DataRecord and a determination of whether it passed the filter or not,
#         construct the resulting RecordSet.
#         """
#         # create new DataRecord
#         data_item = {field: answer[field] for field in self.output_schema.model_fields if field in answer}
#         dr = DataRecord.from_parent(self.output_schema, data_item, parent_record=candidate)

#         # create RecordOpStats object
#         record_op_stats = RecordOpStats(
#             record_id=dr._id,
#             record_parent_ids=dr._parent_ids,
#             record_source_indices=dr._source_indices,
#             record_state=dr.to_dict(include_bytes=False),
#             full_op_id=self.get_full_op_id(),
#             logical_op_id=self.logical_op_id,
#             op_name=self.op_name(),
#             time_per_record=total_time,
#             cost_per_record=generation_stats.cost_per_record,
#             model_name=self.get_model_name(),
#             total_input_tokens=generation_stats.total_input_tokens,
#             total_output_tokens=generation_stats.total_output_tokens,
#             total_input_cost=generation_stats.total_input_cost,
#             total_output_cost=generation_stats.total_output_cost,
#             llm_call_duration_secs=generation_stats.llm_call_duration_secs,
#             fn_call_duration_secs=generation_stats.fn_call_duration_secs,
#             total_llm_calls=generation_stats.total_llm_calls,
#             total_embedding_llm_calls=generation_stats.total_embedding_llm_calls,
#             answer={k: v.description if isinstance(v, Context) else v for k, v in answer.items()},
#             op_details={k: str(v) for k, v in self.get_id_params().items()},
#         )

#         return DataRecordSet([dr], [record_op_stats])

#     def __call__(self, candidate: DataRecord) -> Any:
#         start_time = time.time()

#         # get the input context object and its tools
#         input_context: Context = candidate.context
#         description = input_context.description
#         tools = [tool(make_tool(f)) for f in input_context.tools]

#         # TODO: add semantic operators to tools
#         data_discovery_agent = DataDiscoveryAgent(self.context_id, description, model=self.model, tools=tools)
#         search_manager_agent = SearchManagerAgent(self.context_id, description, model=self.model, tools=tools, managed_agents=[data_discovery_agent])

#         # perform the computation
#         result = search_manager_agent.run(self.search_query)

#         # compute generation stats
#         response = result.output
#         input_tokens = result.token_usage.input_tokens
#         output_tokens = result.token_usage.output_tokens
#         cost_per_input_token = (3.0 / 1e6) if "anthropic" in self.model_id else (0.15 / 1e6) # (15.0 / 1e6)
#         cost_per_output_token = (15.0 / 1e6) if "anthropic" in self.model_id else (0.6 / 1e6) # (60.0 / 1e6)
#         input_cost = input_tokens * cost_per_input_token
#         output_cost = output_tokens * cost_per_output_token
#         generation_stats = GenerationStats(
#             model_name=self.model_id,
#             total_input_tokens=input_tokens,
#             total_output_tokens=output_tokens,
#             total_input_cost=input_cost,
#             total_output_cost=output_cost,
#             cost_per_record=input_cost + output_cost,
#             llm_call_duration_secs=time.time() - start_time,
#         )

#         # update the description of the Context to include the search result
#         new_description = f"RESULT: {response}\n\n"
#         cm = ContextManager()
#         cm.update_context(id=self.context_id, description=new_description)

#         # create and return record set
#         field_answers = {
#             "context": cm.get_context(id=self.context_id),
#         }
#         record_set = self._create_record_set(
#             candidate,
#             generation_stats,
#             time.time() - start_time,
#             field_answers,
#         )

#         return record_set
