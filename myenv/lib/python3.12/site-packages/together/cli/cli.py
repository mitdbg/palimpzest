from __future__ import annotations

import os
from typing import Any

import click

import together
from together.cli.api.chat import chat, interactive
from together.cli.api.completions import completions
from together.cli.api.files import files
from together.cli.api.finetune import fine_tuning
from together.cli.api.images import images
from together.cli.api.models import models
from together.constants import MAX_RETRIES, TIMEOUT_SECS


def print_version(ctx: click.Context, params: Any, value: Any) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.echo(f"Version {together.version}")
    ctx.exit()


@click.group()
@click.pass_context
@click.option(
    "--api-key",
    type=str,
    help="API Key. Defaults to environment variable `TOGETHER_API_KEY`",
    default=os.getenv("TOGETHER_API_KEY"),
)
@click.option(
    "--base-url", type=str, help="API Base URL. Defaults to Together AI endpoint."
)
@click.option(
    "--timeout", type=int, help=f"Request timeout. Defaults to {TIMEOUT_SECS} seconds"
)
@click.option(
    "--max-retries",
    type=int,
    help=f"Maximum number of HTTP retries. Defaults to {MAX_RETRIES}.",
)
@click.option(
    "--version",
    is_flag=True,
    callback=print_version,
    expose_value=False,
    is_eager=True,
    help="Print version",
)
@click.option("--debug", help="Debug mode", is_flag=True)
def main(
    ctx: click.Context,
    api_key: str | None,
    base_url: str | None,
    timeout: int | None,
    max_retries: int | None,
    debug: bool | None,
) -> None:
    """This is a sample CLI tool."""
    together.log = "debug" if debug else None
    ctx.obj = together.Together(
        api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries
    )


main.add_command(chat)
main.add_command(interactive)
main.add_command(completions)
main.add_command(images)
main.add_command(files)
main.add_command(fine_tuning)
main.add_command(models)

if __name__ == "__main__":
    main()
