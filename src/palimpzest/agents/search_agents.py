import json
import textwrap
import time
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from rich.console import Group
from rich.live import Live
from rich.markdown import Markdown
from rich.rule import Rule
from rich.text import Text

if TYPE_CHECKING:
    import PIL.Image

from smolagents.agent_types import handle_agent_output_types
from smolagents.agents import (
    ActionOutput,
    CodeAgent,
    FinalAnswerPromptTemplate,
    ManagedAgentPromptTemplate,
    PlanningPromptTemplate,
    PromptTemplates,
    RunResult,
    ToolOutput,
    populate_template,
)
from smolagents.local_python_executor import fix_final_answer_code
from smolagents.memory import (
    ActionStep,
    FinalAnswerStep,
    PlanningStep,
    SystemPromptStep,
    TaskStep,
    Timing,
    TokenUsage,
    ToolCall,
)
from smolagents.models import (
    CODEAGENT_RESPONSE_FORMAT,
    ChatMessage,
    ChatMessageStreamDelta,
    MessageRole,
    agglomerate_stream_deltas,
)
from smolagents.monitoring import YELLOW_HEX, LogLevel
from smolagents.utils import (
    AgentError,
    AgentExecutionError,
    AgentGenerationError,
    AgentMaxStepsError,
    AgentParsingError,
    extract_code_from_text,
    parse_code_blobs,
    truncate_content,
)

from palimpzest.prompts import (
    CODE_AGENT_SYSTEM_PROMPT,
    DATA_DISCOVERY_AGENT_INITIAL_PLAN_PROMPT,
    DATA_DISCOVERY_AGENT_REPORT_PROMPT,
    DATA_DISCOVERY_AGENT_TASK_PROMPT,
    DATA_DISCOVERY_AGENT_UPDATE_PLAN_POST_MESSAGES_PROMPT,
    DATA_DISCOVERY_AGENT_UPDATE_PLAN_PRE_MESSAGES_PROMPT,
    FINAL_ANSWER_POST_MESSAGES_PROMPT,
    FINAL_ANSWER_PRE_MESSAGES_PROMPT,
)


# TODO: make this use memory the way you want
class PZBaseAgent(CodeAgent):
    def __init__(self, run_id: str, context_description: str, *args, **kwargs):
        # memory_config = {
        #     "vector_store": {
        #         "provider": "chroma",
        #         "config": {
        #             "collection_name": f"palimpzest-memory-{self.__class__.__name__}",
        #             "path": "./pz-chroma",
        #         }
        #     }
        # }
        # self.pz_memory = Memory.from_config(memory_config)
        self.run_id = run_id
        self.context_description = context_description
        super().__init__(*args, **kwargs)

    def write_memory_to_messages(
        self,
        summary_mode: bool = False,
    ) -> list[ChatMessage]:
        """
        Reads past llm_outputs, actions, and observations or errors from the memory into a series of messages
        that can be used as input to the LLM. Adds a number of keywords (such as PLAN, error, etc) to help
        the LLM.
        """
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for memory_step in self.memory.steps:
            messages.extend(memory_step.to_messages(summary_mode=summary_mode))
        return messages

    def _generate_planning_step(
        self, task, is_first_step: bool, step: int
    ) -> Generator[ChatMessageStreamDelta | PlanningStep]:
        start_time = time.time()
        if is_first_step:
            input_messages = [
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "text",
                            "text": populate_template(
                                self.prompt_templates["planning"]["initial_plan"],
                                variables={"task": task, "tools": self.tools, "managed_agents": self.managed_agents, "context_description": self.context_description},
                            ),
                        }
                    ],
                )
            ]
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                output_stream = self.model.generate_stream(input_messages, stop_sequences=["<end_plan>"])  # type: ignore
                input_tokens, output_tokens = 0, 0
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        if event.content is not None:
                            plan_message_content += event.content
                            live.update(Markdown(plan_message_content))
                            if event.token_usage:
                                output_tokens += event.token_usage.output_tokens
                                input_tokens = event.token_usage.input_tokens
                        yield event
            else:
                plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                plan_message_content = plan_message.content
                input_tokens, output_tokens = (
                    (
                        plan_message.token_usage.input_tokens,
                        plan_message.token_usage.output_tokens,
                    )
                    if plan_message.token_usage
                    else (None, None)
                )
            plan = textwrap.dedent(
                f"""Here are the facts I know and the plan of action that I will follow to solve the task:\n```\n{plan_message_content}\n```"""
            )
        else:
            # Summary mode removes the system prompt and previous planning messages output by the model.
            # Removing previous planning messages avoids influencing too much the new plan.
            memory_messages = self.write_memory_to_messages(summary_mode=True)
            plan_update_pre = ChatMessage(
                role=MessageRole.SYSTEM,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_pre_messages"], variables={"task": task, "context_description": self.context_description}
                        ),
                    }
                ],
            )
            plan_update_post = ChatMessage(
                role=MessageRole.USER,
                content=[
                    {
                        "type": "text",
                        "text": populate_template(
                            self.prompt_templates["planning"]["update_plan_post_messages"],
                            variables={
                                "task": task,
                                "tools": self.tools,
                                "managed_agents": self.managed_agents,
                                "remaining_steps": (self.max_steps - step),
                                "context_description": self.context_description,
                            },
                        ),
                    }
                ],
            )
            input_messages = [plan_update_pre] + memory_messages + [plan_update_post]
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                plan_message_content = ""
                input_tokens, output_tokens = 0, 0
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in self.model.generate_stream(
                        input_messages,
                        stop_sequences=["<end_plan>"],
                    ):  # type: ignore
                        if event.content is not None:
                            plan_message_content += event.content
                            live.update(Markdown(plan_message_content))
                            if event.token_usage:
                                output_tokens += event.token_usage.output_tokens
                                input_tokens = event.token_usage.input_tokens
                        yield event
            else:
                plan_message = self.model.generate(input_messages, stop_sequences=["<end_plan>"])
                plan_message_content = plan_message.content
                if plan_message.token_usage is not None:
                    input_tokens, output_tokens = (
                        plan_message.token_usage.input_tokens,
                        plan_message.token_usage.output_tokens,
                    )
            plan = textwrap.dedent(
                f"""I still need to solve the task I was given:\n```\n{self.task}\n```\n\nHere are the facts I know and my new/updated plan of action to solve the task:\n```\n{plan_message_content}\n```"""
            )
        log_headline = "Initial plan" if is_first_step else "Updated plan"
        self.logger.log(Rule(f"[bold]{log_headline}", style="orange"), Text(plan), level=LogLevel.INFO)
        yield PlanningStep(
            model_input_messages=input_messages,
            plan=plan,
            model_output_message=ChatMessage(role=MessageRole.ASSISTANT, content=plan_message_content),
            token_usage=TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens),
            timing=Timing(start_time=start_time, end_time=time.time()),
        )

    # def _curate_messages(self, input_messages: list[ChatMessage]) -> list[ChatMessage]:
    #     """
    #     Try returning:
    #     - System Prompt + task
    #     - Current Plan
    #     - Summary of previous conversation
    #     """
    #     # initialize with the system prompt & original task
    #     curated_messages = input_messages[:2]

    #     # find the last planning step message
    #     idx = len(self.memory.steps) - 1
    #     while idx > -1:
    #         step = self.memory.steps[idx]
    #         if isinstance(step, PlanningStep):
    #             curated_messages.append(step.model_output_message)
    #             break
    #         idx -= 1

    #     # add summary of chat history
    #     history = self.pz_memory.search("A condensed summary of the execution trace of the agent.", run_id=self.run_id)
    #     for msg in history["results"]:
    #         pass

    #     return curated_messages

    def _step_stream(self, memory_step: ActionStep) -> Generator[ChatMessageStreamDelta | ActionOutput | ToolOutput]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()

        ### Generate model output ###
        memory_step.model_input_messages = input_messages
        try:
            additional_args: dict[str, Any] = {}
            if self.grammar:
                additional_args["grammar"] = self.grammar
            if self._use_structured_outputs_internally:
                additional_args["response_format"] = CODEAGENT_RESPONSE_FORMAT
            if self.stream_outputs:
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=["<end_code>", "Observation:", "Calling tools:"],
                    **additional_args,
                )
                chat_message_stream_deltas: list[ChatMessageStreamDelta] = []
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        chat_message_stream_deltas.append(event)
                        live.update(
                            Markdown(agglomerate_stream_deltas(chat_message_stream_deltas).render_as_markdown())
                        )
                        yield event
                chat_message = agglomerate_stream_deltas(chat_message_stream_deltas)
                memory_step.model_output_message = chat_message
                output_text = chat_message.content
            else:
                chat_message: ChatMessage = self.model.generate(
                    input_messages,
                    stop_sequences=["<end_code>", "Observation:", "Calling tools:"],
                    **additional_args,
                )
                memory_step.model_output_message = chat_message
                output_text = chat_message.content
                self.logger.log_markdown(
                    content=output_text,
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            # This adds <end_code> sequence to the history.
            # This will nudge ulterior LLM calls to finish with <end_code>, thus efficiently stopping generation.
            if output_text and output_text.strip().endswith("```"):
                output_text += "<end_code>"
                memory_step.model_output_message.content = output_text

            memory_step.token_usage = chat_message.token_usage
            memory_step.model_output = output_text
        except Exception as e:
            raise AgentGenerationError(f"Error in generating model output:\n{e}", self.logger) from e

        ### Parse output ###
        try:
            if self._use_structured_outputs_internally:
                code_action = json.loads(output_text)["code"]
                code_action = extract_code_from_text(code_action) or code_action
            else:
                code_action = parse_code_blobs(output_text)
            code_action = fix_final_answer_code(code_action)
            memory_step.code_action = code_action
        except Exception as e:
            error_msg = f"Error in code parsing:\n{e}\nMake sure to provide correct code blobs."
            raise AgentParsingError(error_msg, self.logger) from e

        memory_step.tool_calls = [
            ToolCall(
                name="python_interpreter",
                arguments=code_action,
                id=f"call_{len(self.memory.steps)}",
            )
        ]

        ### Execute action ###
        self.logger.log_code(title="Executing parsed code:", content=code_action, level=LogLevel.INFO)
        is_final_answer = False
        try:
            output, execution_logs, is_final_answer = self.python_executor(code_action)
            execution_outputs_console = []
            if len(execution_logs) > 0:
                execution_outputs_console += [
                    Text("Execution logs:", style="bold"),
                    Text(execution_logs),
                ]
            observation = "Execution logs:\n" + execution_logs
        except Exception as e:
            if hasattr(self.python_executor, "state") and "_print_outputs" in self.python_executor.state:
                execution_logs = str(self.python_executor.state["_print_outputs"])
                if len(execution_logs) > 0:
                    execution_outputs_console = [
                        Text("Execution logs:", style="bold"),
                        Text(execution_logs),
                    ]
                    memory_step.observations = "Execution logs:\n" + execution_logs
                    self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
            error_msg = str(e)
            if "Import of " in error_msg and " is not allowed" in error_msg:
                self.logger.log(
                    "[bold red]Warning to user: Code execution failed due to an unauthorized import - Consider passing said import under `additional_authorized_imports` when initializing your CodeAgent.",
                    level=LogLevel.INFO,
                )
            raise AgentExecutionError(error_msg, self.logger) from e

        truncated_output = truncate_content(str(output))
        observation += "Last output from code snippet:\n" + truncated_output
        memory_step.observations = observation
        
        # # TODO: add output to self.pz_memory
        # def get_role(msg_role):
        #     return str(msg_role).split(".")[-1].lower()

        # messages = [
        #     {"role": get_role(memory_step.model_output_message.role), "content": memory_step.model_output_message.content},
        #     {"role": "user", "content": memory_step.observations},
        # ]
        # self.pz_memory.add(messages, run_id=self.run_id, agent_id=self.name)

        execution_outputs_console += [
            Text(
                f"{('Out - Final answer' if is_final_answer else 'Out')}: {truncated_output}",
                style=(f"bold {YELLOW_HEX}" if is_final_answer else ""),
            ),
        ]
        self.logger.log(Group(*execution_outputs_console), level=LogLevel.INFO)
        memory_step.action_output = output
        yield ActionOutput(output=output, is_final_answer=is_final_answer)

    def _run_stream(
        self, task: str, max_steps: int, images: list["PIL.Image.Image"] | None = None
    ) -> Generator[ActionStep | PlanningStep | FinalAnswerStep | ChatMessageStreamDelta]:
        """
        Execute the agent.
        """
        self.step_number = 1
        returned_final_answer = False
        while not returned_final_answer and self.step_number <= self.max_steps:
            # Run a planning step if scheduled
            if self.planning_interval is not None and (
                self.step_number == 1 or (self.step_number - 1) % self.planning_interval == 0
            ):
                planning_start_time = time.time()
                planning_step = None
                for element in self._generate_planning_step(
                    self.task, is_first_step=len(self.memory.steps) == 1, step=self.step_number
                ):  # Don't use the attribute step_number here, because there can be steps from previous runs
                    yield element
                    planning_step = element
                assert isinstance(planning_step, PlanningStep)  # Last yielded element should be a PlanningStep
                self.memory.steps.append(planning_step)
                planning_end_time = time.time()
                planning_step.timing = Timing(
                    start_time=planning_start_time,
                    end_time=planning_end_time,
                )

            # Start action step!
            action_step_start_time = time.time()
            action_step = ActionStep(
                step_number=self.step_number,
                timing=Timing(start_time=action_step_start_time),
                observations_images=images,
            )
            self.logger.log_rule(f"Step {self.step_number}", level=LogLevel.INFO)
            try:
                for output in self._step_stream(action_step):
                    # Yield streaming deltas
                    if not isinstance(output, (ActionOutput, ToolOutput)):
                        # non-action, non-tool output
                        yield output

                    if isinstance(output, (ActionOutput, ToolOutput)) and output.is_final_answer:
                        if self.final_answer_checks:
                            self._validate_final_answer(output.output)
                        returned_final_answer = True
                        action_step.is_final_answer = True
                        final_answer = output.output
                        # handle final step
            except AgentGenerationError as e:
                # Agent generation errors are not caused by a Model error but an implementation error: so we should raise them and exit.
                raise e
            except AgentError as e:
                # Other AgentError types are caused by the Model, so we should log them and iterate.
                action_step.error = e
            finally:
                self._finalize_step(action_step)
                self.memory.steps.append(action_step)
                yield action_step
                self.step_number += 1

        if not returned_final_answer and self.step_number == self.max_steps + 1:
            final_answer = self._handle_max_steps_reached(self.task, images)
            yield action_step
        yield FinalAnswerStep(handle_agent_output_types(final_answer))

    def run(
        self,
        task: str,
        stream: bool = False,
        reset: bool = True,
        images: list["PIL.Image.Image"] | None = None,
        additional_args: dict | None = None,
        max_steps: int | None = None,
    ):
        """
        Run the agent for the given task.

        Args:
            task (`str`): Task to perform.
            stream (`bool`): Whether to run in streaming mode.
                If `True`, returns a generator that yields each step as it is executed. You must iterate over this generator to process the individual steps (e.g., using a for loop or `next()`).
                If `False`, executes all steps internally and returns only the final answer after completion.
            reset (`bool`): Whether to reset the conversation or keep it going from previous run.
            images (`list[PIL.Image.Image]`, *optional*): Image(s) objects.
            additional_args (`dict`, *optional*): Any other variables that you want to pass to the agent run, for instance images or dataframes. Give them clear names!
            max_steps (`int`, *optional*): Maximum number of steps the agent can take to solve the task. if not provided, will use the agent's default value.

        Example:
        ```py
        from smolagents import CodeAgent
        agent = CodeAgent(tools=[])
        agent.run("What is the result of 2 power 3.7384?")
        ```
        """
        max_steps = max_steps or self.max_steps
        self.task = task
        self.interrupt_switch = False
        if additional_args is not None:
            self.state.update(additional_args)
            self.task += f"""
You have been provided with these additional arguments, that you can access using the keys as variables in your python code:
{str(additional_args)}."""

        self.memory.system_prompt = SystemPromptStep(system_prompt=self.system_prompt)
        if reset:
            self.memory.reset()
            self.monitor.reset()

        self.logger.log_task(
            content=self.task.strip(),
            subtitle=f"{type(self.model).__name__} - {(self.model.model_id if hasattr(self.model, 'model_id') else '')}",
            level=LogLevel.INFO,
            title=self.name if hasattr(self, "name") else None,
        )
        self.memory.steps.append(TaskStep(task=self.task, task_images=images))

        if getattr(self, "python_executor", None):
            self.python_executor.send_variables(variables=self.state)
            self.python_executor.send_tools({**self.tools, **self.managed_agents})

        if stream:
            # The steps are returned as they are executed through a generator to iterate on.
            return self._run_stream(task=self.task, max_steps=max_steps, images=images)
        run_start_time = time.time()
        # Outputs are returned only at the end. We only look at the last step.

        steps = list(self._run_stream(task=self.task, max_steps=max_steps, images=images))
        assert isinstance(steps[-1], FinalAnswerStep)
        output = steps[-1].output

        if self.return_full_result:
            total_input_tokens = 0
            total_output_tokens = 0
            correct_token_usage = True
            for step in self.memory.steps:
                if isinstance(step, (ActionStep, PlanningStep)):
                    if step.token_usage is None:
                        correct_token_usage = False
                        break
                    else:
                        total_input_tokens += step.token_usage.input_tokens
                        total_output_tokens += step.token_usage.output_tokens
            if correct_token_usage:
                token_usage = TokenUsage(input_tokens=total_input_tokens, output_tokens=total_output_tokens)
            else:
                token_usage = None

            if self.memory.steps and isinstance(getattr(self.memory.steps[-1], "error", None), AgentMaxStepsError):
                state = "max_steps_error"
            else:
                state = "success"

            messages = self.memory.get_full_steps()

            return RunResult(
                output=output,
                token_usage=token_usage,
                messages=messages,
                timing=Timing(start_time=run_start_time, end_time=time.time()),
                state=state,
            )

        return output


class PZBaseManagedAgent(PZBaseAgent):

    def __call__(self, task: str, **kwargs):
        """Adds additional prompting for the managed agent, runs it, and wraps the output.
        This method is called only by a managed agent.
        """
        full_task = populate_template(
            self.prompt_templates["managed_agent"]["task"],
            variables=dict(name=self.name, task=task, context_description=self.context_description),
        )
        result = self.run(full_task, **kwargs)
        report = result.output if isinstance(result, RunResult) else result
        answer = populate_template(
            self.prompt_templates["managed_agent"]["report"], variables=dict(name=self.name, final_answer=report)
        )
        if self.provide_run_summary:
            answer += "\n\nFor more detail, find below a summary of this agent's work:\n<summary_of_work>\n"
            for message in self.write_memory_to_messages(summary_mode=True):
                content = message.content
                answer += "\n" + truncate_content(str(content)) + "\n---"
            answer += "\n</summary_of_work>"
        return answer


class DataDiscoveryAgent(PZBaseManagedAgent):
    def __init__(self, run_id: str, context_description: str, *args, **kwargs):
        self.description = """A team member that will search a data repository to find files which help to answer your question.
    Ask him for all your questions that require searching a repository of relevant data.
    Provide him as much context as possible, in particular if you need to search on a specific timeframe!
    And don't hesitate to provide him with a complex search task, like finding a difference between two files.
    Your request must be a real sentence, not a keyword search! Like "Find me this information (...)" rather than a few keywords.
        """
        prompt_templates = PromptTemplates(
            system_prompt=CODE_AGENT_SYSTEM_PROMPT,
            planning=PlanningPromptTemplate(
                initial_plan=DATA_DISCOVERY_AGENT_INITIAL_PLAN_PROMPT,
                update_plan_pre_messages=DATA_DISCOVERY_AGENT_UPDATE_PLAN_PRE_MESSAGES_PROMPT,
                update_plan_post_messages=DATA_DISCOVERY_AGENT_UPDATE_PLAN_POST_MESSAGES_PROMPT,
            ),
            managed_agent=ManagedAgentPromptTemplate(task=DATA_DISCOVERY_AGENT_TASK_PROMPT, report=DATA_DISCOVERY_AGENT_REPORT_PROMPT),
            final_answer=FinalAnswerPromptTemplate(pre_messages=FINAL_ANSWER_PRE_MESSAGES_PROMPT, post_messages=FINAL_ANSWER_POST_MESSAGES_PROMPT),
        )

        super().__init__(
            *args,
            run_id=run_id,
            context_description=context_description,
            prompt_templates=prompt_templates,
            max_steps=20,
            verbosity_level=2,
            planning_interval=4,
            name="data_discovery_agent",
            description=self.description,
            provide_run_summary=True,
            **kwargs,
        )
        self.prompt_templates["managed_agent"]["task"] += """Additionally, if after some searching you find out that you need more information to answer the question, you can use `final_answer` with your request for clarification as argument to request for more information."""


class SearchManagerAgent(PZBaseAgent):
    def __init__(self, run_id: str, context_description: str, *args, **kwargs):
        prompt_templates = PromptTemplates(
            system_prompt=CODE_AGENT_SYSTEM_PROMPT,
            planning=PlanningPromptTemplate(
                initial_plan=DATA_DISCOVERY_AGENT_INITIAL_PLAN_PROMPT,
                update_plan_pre_messages=DATA_DISCOVERY_AGENT_UPDATE_PLAN_PRE_MESSAGES_PROMPT,
                update_plan_post_messages=DATA_DISCOVERY_AGENT_UPDATE_PLAN_POST_MESSAGES_PROMPT,
            ),
            managed_agent=ManagedAgentPromptTemplate(task=DATA_DISCOVERY_AGENT_TASK_PROMPT, report=DATA_DISCOVERY_AGENT_REPORT_PROMPT),
            final_answer=FinalAnswerPromptTemplate(pre_messages=FINAL_ANSWER_PRE_MESSAGES_PROMPT, post_messages=FINAL_ANSWER_POST_MESSAGES_PROMPT),
        )
        super().__init__(
            *args,
            run_id=run_id,
            context_description=context_description,
            prompt_templates=prompt_templates,
            max_steps=12,
            verbosity_level=2,
            additional_authorized_imports=["*"],
            planning_interval=4,
            return_full_result=True,
            **kwargs,
        )

# class ManagerAgent(CodeAgent):

#     def _step_stream(self, memory_step: ActionStep) -> Generator[ChatMessageStreamDelta | ActionOutput | ToolOutput]:
#         """
#         Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
#         Yields ChatMessageStreamDelta during the run if streaming is enabled.
#         At the end, yields either None if the step is not final, or the final answer.
#         """
#         raise NotImplementedError("This method should be implemented in child classes")