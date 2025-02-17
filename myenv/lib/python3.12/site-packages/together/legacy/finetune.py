from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal

import together
from together.legacy.base import API_KEY_WARNING, deprecated


class Finetune:
    @classmethod
    @deprecated  # type: ignore
    def create(
        cls,
        training_file: str,  # training file_id
        model: str,
        n_epochs: int = 1,
        n_checkpoints: int | None = 1,
        batch_size: int | None = 32,
        learning_rate: float = 0.00001,
        suffix: (
            str | None
        ) = None,  # resulting finetuned model name will include the suffix
        estimate_price: bool = False,
        wandb_api_key: str | None = None,
        confirm_inputs: bool = False,
    ):
        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        if estimate_price:
            raise ValueError("Price estimation is not supported in version >= 1.0.1")

        if confirm_inputs:
            raise ValueError("Input confirmation is not supported in version >= 1.0.1")

        client = together.Together(api_key=api_key)

        return client.fine_tuning.create(
            training_file=training_file,
            model=model,
            n_epochs=n_epochs,
            n_checkpoints=n_checkpoints,
            batch_size=batch_size if isinstance(batch_size, int) else "max",
            learning_rate=learning_rate,
            suffix=suffix,
            wandb_api_key=wandb_api_key,
        ).model_dump(exclude_none=True)

    @classmethod
    @deprecated  # type: ignore
    def list(
        cls,
    ) -> Dict[str, Any]:
        """Legacy finetuning list function."""

        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        client = together.Together(api_key=api_key)

        return client.fine_tuning.list().model_dump(exclude_none=True)

    @classmethod
    @deprecated  # type: ignore
    def retrieve(
        cls,
        fine_tune_id: str,
    ) -> Dict[str, Any]:
        """Legacy finetuning retrieve function."""

        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        client = together.Together(api_key=api_key)

        return client.fine_tuning.retrieve(id=fine_tune_id).model_dump(
            exclude_none=True
        )

    @classmethod
    @deprecated  # type: ignore
    def cancel(
        cls,
        fine_tune_id: str,
    ) -> Dict[str, Any]:
        """Legacy finetuning cancel function."""

        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        client = together.Together(api_key=api_key)

        return client.fine_tuning.cancel(id=fine_tune_id).model_dump(exclude_none=True)

    @classmethod
    @deprecated  # type: ignore
    def list_events(
        cls,
        fine_tune_id: str,
    ) -> Dict[str, Any]:
        """Legacy finetuning list events function."""

        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        client = together.Together(api_key=api_key)

        return client.fine_tuning.list_events(id=fine_tune_id).model_dump(
            exclude_none=True
        )

    @classmethod
    @deprecated  # type: ignore
    def get_checkpoints(
        cls,
        fine_tune_id: str,
    ) -> List[Any]:
        """Legacy finetuning get checkpoints function."""

        finetune_events = list(cls.retrieve(fine_tune_id=fine_tune_id)["events"])

        saved_events = [i for i in finetune_events if i["type"] in ["CHECKPOINT_SAVE"]]

        return saved_events

    @classmethod
    @deprecated  # type: ignore
    def get_job_status(cls, fine_tune_id: str) -> str:
        """Legacy finetuning get job status function."""
        return str(cls.retrieve(fine_tune_id=fine_tune_id)["status"])

    @classmethod
    @deprecated  # type: ignore
    def is_final_model_available(cls, fine_tune_id: str) -> bool:
        """Legacy finetuning is final model available function."""

        finetune_events = list(cls.retrieve(fine_tune_id=fine_tune_id)["events"])

        for i in finetune_events:
            if i["type"] in ["JOB_COMPLETE", "JOB_ERROR"]:
                if i["checkpoint_path"] != "":
                    return False
                else:
                    return True
        return False

    @classmethod
    @deprecated  # type: ignore
    def download(
        cls,
        fine_tune_id: str,
        output: str | None = None,
        step: int = -1,
    ) -> Dict[str, Any]:
        """Legacy finetuning download function."""

        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        client = together.Together(api_key=api_key)

        return client.fine_tuning.download(
            id=fine_tune_id, output=output, checkpoint_step=step
        ).model_dump(exclude_none=True)
