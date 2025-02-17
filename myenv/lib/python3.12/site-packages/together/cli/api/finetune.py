from __future__ import annotations

import json
from datetime import datetime
from textwrap import wrap
from typing import Any, Literal

import click
from click.core import ParameterSource  # type: ignore[attr-defined]
from rich import print as rprint
from tabulate import tabulate

from together import Together
from together.cli.api.utils import INT_WITH_MAX
from together.utils import finetune_price_to_dollars, log_warn, parse_timestamp
from together.types.finetune import DownloadCheckpointType, FinetuneTrainingLimits


_CONFIRMATION_MESSAGE = (
    "You are about to create a fine-tuning job. "
    "The cost of your job will be determined by the model size, the number of tokens "
    "in the training file, the number of tokens in the validation file, the number of epochs, and "
    "the number of evaluations. Visit https://www.together.ai/pricing to get a price estimate.\n"
    "You can pass `-y` or `--confirm` to your command to skip this message.\n\n"
    "Do you want to proceed?"
)


class DownloadCheckpointTypeChoice(click.Choice):
    def __init__(self) -> None:
        super().__init__([ct.value for ct in DownloadCheckpointType])

    def convert(
        self, value: str, param: click.Parameter | None, ctx: click.Context | None
    ) -> DownloadCheckpointType:
        value = super().convert(value, param, ctx)
        return DownloadCheckpointType(value)


@click.group(name="fine-tuning")
@click.pass_context
def fine_tuning(ctx: click.Context) -> None:
    """Fine-tunes API commands"""
    pass


@fine_tuning.command()
@click.pass_context
@click.option(
    "--training-file", type=str, required=True, help="Training file ID from Files API"
)
@click.option("--model", type=str, required=True, help="Base model name")
@click.option("--n-epochs", type=int, default=1, help="Number of epochs to train for")
@click.option(
    "--validation-file", type=str, default="", help="Validation file ID from Files API"
)
@click.option("--n-evals", type=int, default=0, help="Number of evaluation loops")
@click.option(
    "--n-checkpoints", type=int, default=1, help="Number of checkpoints to save"
)
@click.option("--batch-size", type=INT_WITH_MAX, default="max", help="Train batch size")
@click.option("--learning-rate", type=float, default=1e-5, help="Learning rate")
@click.option(
    "--lora/--no-lora",
    type=bool,
    default=False,
    help="Whether to use LoRA adapters for fine-tuning",
)
@click.option("--lora-r", type=int, default=8, help="LoRA adapters' rank")
@click.option("--lora-dropout", type=float, default=0, help="LoRA adapters' dropout")
@click.option("--lora-alpha", type=float, default=8, help="LoRA adapters' alpha")
@click.option(
    "--lora-trainable-modules",
    type=str,
    default="all-linear",
    help="Trainable modules for LoRA adapters. For example, 'all-linear', 'q_proj,v_proj'",
)
@click.option(
    "--suffix", type=str, default=None, help="Suffix for the fine-tuned model name"
)
@click.option("--wandb-api-key", type=str, default=None, help="Wandb API key")
@click.option(
    "--confirm",
    "-y",
    type=bool,
    is_flag=True,
    default=False,
    help="Whether to skip the launch confirmation message",
)
def create(
    ctx: click.Context,
    training_file: str,
    validation_file: str,
    model: str,
    n_epochs: int,
    n_evals: int,
    n_checkpoints: int,
    batch_size: int | Literal["max"],
    learning_rate: float,
    lora: bool,
    lora_r: int,
    lora_dropout: float,
    lora_alpha: float,
    lora_trainable_modules: str,
    suffix: str,
    wandb_api_key: str,
    confirm: bool,
) -> None:
    """Start fine-tuning"""
    client: Together = ctx.obj

    training_args: dict[str, Any] = dict(
        training_file=training_file,
        model=model,
        n_epochs=n_epochs,
        validation_file=validation_file,
        n_evals=n_evals,
        n_checkpoints=n_checkpoints,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora=lora,
        lora_r=lora_r,
        lora_dropout=lora_dropout,
        lora_alpha=lora_alpha,
        lora_trainable_modules=lora_trainable_modules,
        suffix=suffix,
        wandb_api_key=wandb_api_key,
    )

    model_limits: FinetuneTrainingLimits = client.fine_tuning.get_model_limits(
        model=model
    )

    if lora:
        if model_limits.lora_training is None:
            raise click.BadParameter(
                f"LoRA fine-tuning is not supported for the model `{model}`"
            )

        default_values = {
            "lora_r": model_limits.lora_training.max_rank,
            "batch_size": model_limits.lora_training.max_batch_size,
            "learning_rate": 1e-3,
        }
        for arg in default_values:
            arg_source = ctx.get_parameter_source("arg")  # type: ignore[attr-defined]
            if arg_source == ParameterSource.DEFAULT:
                training_args[arg] = default_values[arg_source]

        if ctx.get_parameter_source("lora_alpha") == ParameterSource.DEFAULT:  # type: ignore[attr-defined]
            training_args["lora_alpha"] = training_args["lora_r"] * 2
    else:
        if model_limits.full_training is None:
            raise click.BadParameter(
                f"Full fine-tuning is not supported for the model `{model}`"
            )

        for param in ["lora_r", "lora_dropout", "lora_alpha", "lora_trainable_modules"]:
            param_source = ctx.get_parameter_source(param)  # type: ignore[attr-defined]
            if param_source != ParameterSource.DEFAULT:
                raise click.BadParameter(
                    f"You set LoRA parameter `{param}` for a full fine-tuning job. "
                    f"Please change the job type with --lora or remove `{param}` from the arguments"
                )

        batch_size_source = ctx.get_parameter_source("batch_size")  # type: ignore[attr-defined]
        if batch_size_source == ParameterSource.DEFAULT:
            training_args["batch_size"] = model_limits.full_training.max_batch_size

    if n_evals <= 0 and validation_file:
        log_warn(
            "Warning: You have specified a validation file but the number of evaluation loops is set to 0. No evaluations will be performed."
        )
    elif n_evals > 0 and not validation_file:
        raise click.BadParameter(
            "You have specified a number of evaluation loops but no validation file."
        )

    if confirm or click.confirm(_CONFIRMATION_MESSAGE, default=True, show_default=True):
        response = client.fine_tuning.create(
            training_file=training_file,
            model=model,
            n_epochs=n_epochs,
            validation_file=validation_file,
            n_evals=n_evals,
            n_checkpoints=n_checkpoints,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lora=lora,
            lora_r=lora_r,
            lora_dropout=lora_dropout,
            lora_alpha=lora_alpha,
            lora_trainable_modules=lora_trainable_modules,
            suffix=suffix,
            wandb_api_key=wandb_api_key,
            verbose=True,
        )

        report_string = f"Successfully submitted a fine-tuning job {response.id}"
        if response.created_at is not None:
            created_time = datetime.strptime(
                response.created_at, "%Y-%m-%dT%H:%M:%S.%f%z"
            )
            # created_at reports UTC time, we use .astimezone() to convert to local time
            formatted_time = created_time.astimezone().strftime("%m/%d/%Y, %H:%M:%S")
            report_string += f" at {formatted_time}"
        rprint(report_string)
    else:
        click.echo("No confirmation received, stopping job launch")


@fine_tuning.command()
@click.pass_context
def list(ctx: click.Context) -> None:
    """List fine-tuning jobs"""
    client: Together = ctx.obj

    response = client.fine_tuning.list()

    response.data = response.data or []

    response.data.sort(key=lambda x: parse_timestamp(x.created_at or ""))

    display_list = []
    for i in response.data:
        display_list.append(
            {
                "Fine-tune ID": i.id,
                "Model Output Name": "\n".join(wrap(i.output_name or "", width=30)),
                "Status": i.status,
                "Created At": i.created_at,
                "Price": f"""${finetune_price_to_dollars(
                    float(str(i.total_price))
                )}""",  # convert to string for mypy typing
            }
        )
    table = tabulate(display_list, headers="keys", tablefmt="grid", showindex=True)

    click.echo(table)


@fine_tuning.command()
@click.pass_context
@click.argument("fine_tune_id", type=str, required=True)
def retrieve(ctx: click.Context, fine_tune_id: str) -> None:
    """Retrieve fine-tuning job details"""
    client: Together = ctx.obj

    response = client.fine_tuning.retrieve(fine_tune_id)

    # remove events from response for cleaner output
    response.events = None

    click.echo(json.dumps(response.model_dump(exclude_none=True), indent=4))


@fine_tuning.command()
@click.pass_context
@click.argument("fine_tune_id", type=str, required=True)
@click.option(
    "--quiet", is_flag=True, help="Do not prompt for confirmation before cancelling job"
)
def cancel(ctx: click.Context, fine_tune_id: str, quiet: bool = False) -> None:
    """Cancel fine-tuning job"""
    client: Together = ctx.obj
    if not quiet:
        confirm_response = input(
            "You will be billed for any completed training steps upon cancellation. "
            f"Do you want to cancel job {fine_tune_id}? [y/N]"
        )
        if "y" not in confirm_response.lower():
            click.echo({"status": "Cancel not submitted"})
            return
    response = client.fine_tuning.cancel(fine_tune_id)

    click.echo(json.dumps(response.model_dump(exclude_none=True), indent=4))


@fine_tuning.command()
@click.pass_context
@click.argument("fine_tune_id", type=str, required=True)
def list_events(ctx: click.Context, fine_tune_id: str) -> None:
    """List fine-tuning events"""
    client: Together = ctx.obj

    response = client.fine_tuning.list_events(fine_tune_id)

    response.data = response.data or []

    display_list = []
    for i in response.data:
        display_list.append(
            {
                "Message": "\n".join(wrap(i.message or "", width=50)),
                "Type": i.type,
                "Created At": parse_timestamp(i.created_at or ""),
                "Hash": i.hash,
            }
        )
    table = tabulate(display_list, headers="keys", tablefmt="grid", showindex=True)

    click.echo(table)


@fine_tuning.command()
@click.pass_context
@click.argument("fine_tune_id", type=str, required=True)
@click.option(
    "--output_dir",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    required=False,
    default=None,
    help="Output directory",
)
@click.option(
    "--checkpoint-step",
    type=int,
    required=False,
    default=-1,
    help="Download fine-tuning checkpoint. Defaults to latest.",
)
@click.option(
    "--checkpoint-type",
    type=DownloadCheckpointTypeChoice(),
    required=False,
    default=DownloadCheckpointType.DEFAULT.value,
    help="Specifies checkpoint type. 'merged' and 'adapter' options work only for LoRA jobs.",
)
def download(
    ctx: click.Context,
    fine_tune_id: str,
    output_dir: str,
    checkpoint_step: int,
    checkpoint_type: DownloadCheckpointType,
) -> None:
    """Download fine-tuning checkpoint"""
    client: Together = ctx.obj

    response = client.fine_tuning.download(
        fine_tune_id,
        output=output_dir,
        checkpoint_step=checkpoint_step,
        checkpoint_type=checkpoint_type,
    )

    click.echo(json.dumps(response.model_dump(exclude_none=True), indent=4))
