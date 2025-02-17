from __future__ import annotations

import cmd
import json
from typing import List, Tuple

import click

from together import Together
from together.types.chat_completions import (
    ChatCompletionChoicesChunk,
    ChatCompletionChunk,
    ChatCompletionResponse,
)


class ChatShell(cmd.Cmd):
    intro = "Type /exit to exit, /help, or /? to list commands.\n"
    prompt = ">>> "

    def __init__(
        self,
        client: Together,
        model: str,
        max_tokens: int | None = None,
        stop: List[str] | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        presence_penalty: float | None = None,
        frequency_penalty: float | None = None,
        min_p: float | None = None,
        safety_model: str | None = None,
        system_message: str | None = None,
    ) -> None:
        super().__init__()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.stop = stop
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.min_p = min_p
        self.safety_model = safety_model
        self.system_message = system_message

        self.messages = (
            [{"role": "system", "content": self.system_message}]
            if self.system_message
            else []
        )

    def precmd(self, line: str) -> str:
        if line.startswith("/"):
            return line[1:]
        else:
            return "say " + line

    def do_say(self, arg: str) -> None:
        self.messages.append({"role": "user", "content": arg})

        output = ""

        for chunk in self.client.chat.completions.create(
            messages=self.messages,
            model=self.model,
            max_tokens=self.max_tokens,
            stop=self.stop,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            min_p=self.min_p,
            safety_model=self.safety_model,
            stream=True,
        ):
            # assertions for type checking
            assert isinstance(chunk, ChatCompletionChunk)
            assert chunk.choices
            assert chunk.choices[0].delta

            token = chunk.choices[0].delta.content

            click.echo(token, nl=False)

            output += token or ""

        click.echo("\n")

        self.messages.append({"role": "assistant", "content": output})

    def do_reset(self, arg: str) -> None:
        self.messages = (
            [{"role": "system", "content": self.system_message}]
            if self.system_message
            else []
        )

    def do_exit(self, arg: str) -> bool:
        return True


@click.command(name="chat.interactive")
@click.pass_context
@click.option("--model", type=str, required=True, help="Model name")
@click.option("--max-tokens", type=int, help="Max tokens to generate")
@click.option(
    "--stop", type=str, multiple=True, help="List of strings to stop generation"
)
@click.option("--temperature", type=float, help="Sampling temperature")
@click.option("--top-p", type=int, help="Top p sampling")
@click.option("--top-k", type=float, help="Top k sampling")
@click.option("--repetition-penalty", type=float, help="Repetition penalty")
@click.option("--presence-penalty", type=float, help="Presence penalty")
@click.option("--frequency-penalty", type=float, help="Frequency penalty")
@click.option("--min-p", type=float, help="Minimum p")
@click.option("--safety-model", type=str, help="Moderation model")
@click.option("--system-message", type=str, help="System message to use for the chat")
def interactive(
    ctx: click.Context,
    model: str,
    max_tokens: int | None = None,
    stop: List[str] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    min_p: float | None = None,
    safety_model: str | None = None,
    system_message: str | None = None,
) -> None:
    """Interactive chat shell"""
    client: Together = ctx.obj

    ChatShell(
        client=client,
        model=model,
        max_tokens=max_tokens,
        stop=stop,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        min_p=min_p,
        safety_model=safety_model,
        system_message=system_message,
    ).cmdloop()


@click.command(name="chat.completions")
@click.pass_context
@click.option(
    "--message",
    type=(str, str),
    multiple=True,
    required=True,
    help="Message to generate chat completions from",
)
@click.option("--model", type=str, required=True, help="Model name")
@click.option("--max-tokens", type=int, help="Max tokens to generate")
@click.option(
    "--stop", type=str, multiple=True, help="List of strings to stop generation"
)
@click.option("--temperature", type=float, help="Sampling temperature")
@click.option("--top-p", type=int, help="Top p sampling")
@click.option("--top-k", type=float, help="Top k sampling")
@click.option("--repetition-penalty", type=float, help="Repetition penalty")
@click.option("--presence-penalty", type=float, help="Presence penalty sampling method")
@click.option(
    "--frequency-penalty", type=float, help="Frequency penalty sampling method"
)
@click.option("--min-p", type=float, help="Min p sampling")
@click.option("--no-stream", is_flag=True, help="Disable streaming")
@click.option("--logprobs", type=int, help="Return logprobs. Only works with --raw.")
@click.option("--echo", is_flag=True, help="Echo prompt. Only works with --raw.")
@click.option("--n", type=int, help="Number of output generations")
@click.option("--safety-model", type=str, help="Moderation model")
@click.option("--raw", is_flag=True, help="Output raw JSON")
def chat(
    ctx: click.Context,
    message: List[Tuple[str, str]],
    model: str,
    max_tokens: int | None = None,
    stop: List[str] | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    repetition_penalty: float | None = None,
    presence_penalty: float | None = None,
    frequency_penalty: float | None = None,
    min_p: float | None = None,
    no_stream: bool = False,
    logprobs: int | None = None,
    echo: bool | None = None,
    n: int | None = None,
    safety_model: str | None = None,
    raw: bool = False,
) -> None:
    """Generate chat completions from messages"""
    client: Together = ctx.obj

    messages = [{"role": msg[0], "content": msg[1]} for msg in message]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        max_tokens=max_tokens,
        stop=stop,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        min_p=min_p,
        stream=not no_stream,
        logprobs=logprobs,
        echo=echo,
        n=n,
        safety_model=safety_model,
    )

    if not no_stream:
        for chunk in response:
            # assertions for type checking
            assert isinstance(chunk, ChatCompletionChunk)
            assert chunk.choices

            if raw:
                click.echo(f"{json.dumps(chunk.model_dump(exclude_none=True))}")
                continue

            should_print_header = len(chunk.choices) > 1
            for stream_choice in sorted(chunk.choices, key=lambda c: c.index):  # type: ignore
                assert isinstance(stream_choice, ChatCompletionChoicesChunk)
                assert stream_choice.delta

                if should_print_header:
                    click.echo(f"\n===== Completion {stream_choice.index} =====\n")
                click.echo(f"{stream_choice.delta.content}", nl=False)

                if should_print_header:
                    click.echo("\n")

        # new line after stream ends
        click.echo("\n")
    else:
        # assertions for type checking
        assert isinstance(response, ChatCompletionResponse)
        assert isinstance(response.choices, list)

        if raw:
            click.echo(
                f"{json.dumps(response.model_dump(exclude_none=True), indent=4)}"
            )
            return

        should_print_header = len(response.choices) > 1
        for i, choice in enumerate(response.choices):
            if should_print_header:
                click.echo(f"===== Completion {i} =====")
            click.echo(choice.message.content)  # type: ignore

            if should_print_header:
                click.echo("\n")
