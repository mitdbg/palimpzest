from __future__ import annotations

import json
from typing import List

import click

from together import Together
from together.types import CompletionChunk
from together.types.completions import CompletionChoicesChunk, CompletionResponse


@click.command()
@click.pass_context
@click.argument("prompt", type=str, required=True)
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
@click.option("--no-stream", is_flag=True, help="Disable streaming")
@click.option("--logprobs", type=int, help="Return logprobs. Only works with --raw.")
@click.option("--echo", is_flag=True, help="Echo prompt. Only works with --raw.")
@click.option("--n", type=int, help="Number of output generations")
@click.option("--safety-model", type=str, help="Moderation model")
@click.option("--raw", is_flag=True, help="Return raw JSON response")
def completions(
    ctx: click.Context,
    prompt: str,
    model: str,
    max_tokens: int | None = 512,
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
    """Generate text completions"""
    client: Together = ctx.obj

    response = client.completions.create(
        model=model,
        prompt=prompt,
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
            assert isinstance(chunk, CompletionChunk)
            assert chunk.choices

            if raw:
                click.echo(f"{json.dumps(chunk.model_dump(exclude_none=True))}")
                continue

            should_print_header = len(chunk.choices) > 1
            for stream_choice in sorted(chunk.choices, key=lambda c: c.index):  # type: ignore
                # assertions for type checking
                assert isinstance(stream_choice, CompletionChoicesChunk)
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
        assert isinstance(response, CompletionResponse)
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
            click.echo(choice.text)

            if should_print_header or not choice.text.endswith("\n"):
                click.echo("\n")
