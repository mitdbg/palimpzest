from __future__ import annotations

from pathlib import Path
from typing import Literal

from rich import print as rprint

from together.abstract import api_requestor
from together.filemanager import DownloadManager
from together.together_response import TogetherResponse
from together.types import (
    FinetuneDownloadResult,
    FinetuneList,
    FinetuneListEvents,
    FinetuneRequest,
    FinetuneResponse,
    FinetuneTrainingLimits,
    FullTrainingType,
    LoRATrainingType,
    TogetherClient,
    TogetherRequest,
    TrainingType,
)
from together.types.finetune import DownloadCheckpointType
from together.utils import log_warn_once, normalize_key


class FineTuning:
    def __init__(self, client: TogetherClient) -> None:
        self._client = client

    def create(
        self,
        *,
        training_file: str,
        model: str,
        n_epochs: int = 1,
        validation_file: str | None = "",
        n_evals: int | None = 0,
        n_checkpoints: int | None = 1,
        batch_size: int | Literal["max"] = "max",
        learning_rate: float | None = 0.00001,
        lora: bool = False,
        lora_r: int | None = None,
        lora_dropout: float | None = 0,
        lora_alpha: float | None = None,
        lora_trainable_modules: str | None = "all-linear",
        suffix: str | None = None,
        wandb_api_key: str | None = None,
        verbose: bool = False,
        model_limits: FinetuneTrainingLimits | None = None,
    ) -> FinetuneResponse:
        """
        Method to initiate a fine-tuning job

        Args:
            training_file (str): File-ID of a file uploaded to the Together API
            model (str): Name of the base model to run fine-tune job on
            n_epochs (int, optional): Number of epochs for fine-tuning. Defaults to 1.
            validation file (str, optional): File ID of a file uploaded to the Together API for validation.
            n_evals (int, optional): Number of evaluation loops to run. Defaults to 0.
            n_checkpoints (int, optional): Number of checkpoints to save during fine-tuning.
                Defaults to 1.
            batch_size (int, optional): Batch size for fine-tuning. Defaults to max.
            learning_rate (float, optional): Learning rate multiplier to use for training
                Defaults to 0.00001.
            lora (bool, optional): Whether to use LoRA adapters. Defaults to True.
            lora_r (int, optional): Rank of LoRA adapters. Defaults to 8.
            lora_dropout (float, optional): Dropout rate for LoRA adapters. Defaults to 0.
            lora_alpha (float, optional): Alpha for LoRA adapters. Defaults to 8.
            lora_trainable_modules (str, optional): Trainable modules for LoRA adapters. Defaults to "all-linear".
            suffix (str, optional): Up to 40 character suffix that will be added to your fine-tuned model name.
                Defaults to None.
            wandb_api_key (str, optional): API key for Weights & Biases integration.
                Defaults to None.
            verbose (bool, optional): whether to print the job parameters before submitting a request.
                Defaults to False.
            model_limits (FinetuneTrainingLimits, optional): Limits for the hyperparameters the model in Fine-tuning.
                Defaults to None.

        Returns:
            FinetuneResponse: Object containing information about fine-tuning job.
        """

        if batch_size == "max":
            log_warn_once(
                "Starting from together>=1.3.0, "
                "the default batch size is set to the maximum allowed value for each model."
            )

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        if model_limits is None:
            model_limits = self.get_model_limits(model=model)

        training_type: TrainingType = FullTrainingType()
        if lora:
            if model_limits.lora_training is None:
                raise ValueError(
                    "LoRA adapters are not supported for the selected model."
                )
            lora_r = (
                lora_r if lora_r is not None else model_limits.lora_training.max_rank
            )
            lora_alpha = lora_alpha if lora_alpha is not None else lora_r * 2
            training_type = LoRATrainingType(
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_trainable_modules=lora_trainable_modules,
            )

            batch_size = (
                batch_size
                if batch_size != "max"
                else model_limits.lora_training.max_batch_size
            )
        else:
            if model_limits.full_training is None:
                raise ValueError(
                    "Full training is not supported for the selected model."
                )
            batch_size = (
                batch_size
                if batch_size != "max"
                else model_limits.full_training.max_batch_size
            )

        finetune_request = FinetuneRequest(
            model=model,
            training_file=training_file,
            validation_file=validation_file,
            n_epochs=n_epochs,
            n_evals=n_evals,
            n_checkpoints=n_checkpoints,
            batch_size=batch_size,
            learning_rate=learning_rate,
            training_type=training_type,
            suffix=suffix,
            wandb_key=wandb_api_key,
        )
        if verbose:
            rprint(
                "Submitting a fine-tuning job with the following parameters:",
                finetune_request,
            )
        parameter_payload = finetune_request.model_dump(exclude_none=True)

        response, _, _ = requestor.request(
            options=TogetherRequest(
                method="POST",
                url="fine-tunes",
                params=parameter_payload,
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)

        return FinetuneResponse(**response.data)

    def list(self) -> FinetuneList:
        """
        Lists fine-tune job history

        Returns:
            FinetuneList: Object containing a list of fine-tune jobs
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        response, _, _ = requestor.request(
            options=TogetherRequest(
                method="GET",
                url="fine-tunes",
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)

        return FinetuneList(**response.data)

    def retrieve(self, id: str) -> FinetuneResponse:
        """
        Retrieves fine-tune job details

        Args:
            id (str): Fine-tune ID to retrieve. A string that starts with `ft-`.

        Returns:
            FinetuneResponse: Object containing information about fine-tuning job.
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        response, _, _ = requestor.request(
            options=TogetherRequest(
                method="GET",
                url=f"fine-tunes/{id}",
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)

        return FinetuneResponse(**response.data)

    def cancel(self, id: str) -> FinetuneResponse:
        """
        Method to cancel a running fine-tuning job

        Args:
            id (str): Fine-tune ID to cancel. A string that starts with `ft-`.

        Returns:
            FinetuneResponse: Object containing information about cancelled fine-tuning job.
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        response, _, _ = requestor.request(
            options=TogetherRequest(
                method="POST",
                url=f"fine-tunes/{id}/cancel",
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)

        return FinetuneResponse(**response.data)

    def list_events(self, id: str) -> FinetuneListEvents:
        """
        Lists events of a fine-tune job

        Args:
            id (str): Fine-tune ID to list events for. A string that starts with `ft-`.

        Returns:
            FinetuneListEvents: Object containing list of fine-tune events
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        response, _, _ = requestor.request(
            options=TogetherRequest(
                method="GET",
                url=f"fine-tunes/{id}/events",
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)

        return FinetuneListEvents(**response.data)

    def download(
        self,
        id: str,
        *,
        output: Path | str | None = None,
        checkpoint_step: int = -1,
        checkpoint_type: DownloadCheckpointType = DownloadCheckpointType.DEFAULT,
    ) -> FinetuneDownloadResult:
        """
        Downloads compressed fine-tuned model or checkpoint to local disk.

        Defaults file location to `$PWD/{model_name}.{extension}`

        Args:
            id (str): Fine-tune ID to download. A string that starts with `ft-`.
            output (pathlib.Path | str, optional): Specifies output file name for downloaded model.
                Defaults to None.
            checkpoint_step (int, optional): Specifies step number for checkpoint to download.
                Defaults to -1 (download the final model)
            checkpoint_type (CheckpointType, optional): Specifies which checkpoint to download.
                Defaults to CheckpointType.DEFAULT.

        Returns:
            FinetuneDownloadResult: Object containing downloaded model metadata
        """

        url = f"finetune/download?ft_id={id}"

        if checkpoint_step > 0:
            url += f"&checkpoint_step={checkpoint_step}"

        ft_job = self.retrieve(id)

        if isinstance(ft_job.training_type, FullTrainingType):
            if checkpoint_type != DownloadCheckpointType.DEFAULT:
                raise ValueError(
                    "Only DEFAULT checkpoint type is allowed for FullTrainingType"
                )
            url += "&checkpoint=modelOutputPath"
        elif isinstance(ft_job.training_type, LoRATrainingType):
            if checkpoint_type == DownloadCheckpointType.DEFAULT:
                checkpoint_type = DownloadCheckpointType.MERGED

            if checkpoint_type == DownloadCheckpointType.MERGED:
                url += f"&checkpoint={DownloadCheckpointType.MERGED.value}"
            elif checkpoint_type == DownloadCheckpointType.ADAPTER:
                url += f"&checkpoint={DownloadCheckpointType.ADAPTER.value}"
            else:
                raise ValueError(
                    f"Invalid checkpoint type for LoRATrainingType: {checkpoint_type}"
                )

        remote_name = ft_job.output_name

        download_manager = DownloadManager(self._client)

        if isinstance(output, str):
            output = Path(output)

        downloaded_filename, file_size = download_manager.download(
            url, output, normalize_key(remote_name or id), fetch_metadata=True
        )

        return FinetuneDownloadResult(
            object="local",
            id=id,
            checkpoint_step=checkpoint_step,
            filename=downloaded_filename,
            size=file_size,
        )

    def get_model_limits(self, *, model: str) -> FinetuneTrainingLimits:
        """
        Requests training limits for a specific model

        Args:
            model_name (str): Name of the model to get limits for

        Returns:
            FinetuneTrainingLimits: Object containing training limits for the model
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        model_limits_response, _, _ = requestor.request(
            options=TogetherRequest(
                method="GET",
                url="fine-tunes/models/limits",
                params={"model_name": model},
            ),
            stream=False,
        )

        model_limits = FinetuneTrainingLimits(**model_limits_response.data)

        return model_limits


class AsyncFineTuning:
    def __init__(self, client: TogetherClient) -> None:
        self._client = client

    async def create(
        self,
        *,
        training_file: str,
        model: str,
        n_epochs: int = 1,
        validation_file: str | None = "",
        n_evals: int = 0,
        n_checkpoints: int | None = 1,
        batch_size: int | None = 32,
        learning_rate: float = 0.00001,
        suffix: str | None = None,
        wandb_api_key: str | None = None,
    ) -> FinetuneResponse:
        """
        Async method to initiate a fine-tuning job

        Args:
            training_file (str): File-ID of a file uploaded to the Together API
            model (str): Name of the base model to run fine-tune job on
            n_epochs (int, optional): Number of epochs for fine-tuning. Defaults to 1.
            validation file (str, optional): File ID of a file uploaded to the Together API for validation.
            n_evals (int, optional): Number of evaluation loops to run. Defaults to 0.
            n_checkpoints (int, optional): Number of checkpoints to save during fine-tuning.
                Defaults to 1.
            batch_size (int, optional): Batch size for fine-tuning. Defaults to 32.
            learning_rate (float, optional): Learning rate multiplier to use for training
                Defaults to 0.00001.
            suffix (str, optional): Up to 40 character suffix that will be added to your fine-tuned model name.
                Defaults to None.
            wandb_api_key (str, optional): API key for Weights & Biases integration.
                Defaults to None.

        Returns:
            FinetuneResponse: Object containing information about fine-tuning job.
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        parameter_payload = FinetuneRequest(
            model=model,
            training_file=training_file,
            validation_file=validation_file,
            n_epochs=n_epochs,
            n_evals=n_evals,
            n_checkpoints=n_checkpoints,
            batch_size=batch_size,
            learning_rate=learning_rate,
            suffix=suffix,
            wandb_key=wandb_api_key,
        ).model_dump(exclude_none=True)

        response, _, _ = await requestor.arequest(
            options=TogetherRequest(
                method="POST",
                url="fine-tunes",
                params=parameter_payload,
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)

        return FinetuneResponse(**response.data)

    async def list(self) -> FinetuneList:
        """
        Async method to list fine-tune job history

        Returns:
            FinetuneList: Object containing a list of fine-tune jobs
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        response, _, _ = await requestor.arequest(
            options=TogetherRequest(
                method="GET",
                url="fine-tunes",
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)

        return FinetuneList(**response.data)

    async def retrieve(self, id: str) -> FinetuneResponse:
        """
        Async method to retrieve fine-tune job details

        Args:
            id (str): Fine-tune ID to retrieve. A string that starts with `ft-`.

        Returns:
            FinetuneResponse: Object containing information about fine-tuning job.
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        response, _, _ = await requestor.arequest(
            options=TogetherRequest(
                method="GET",
                url=f"fine-tunes/{id}",
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)

        return FinetuneResponse(**response.data)

    async def cancel(self, id: str) -> FinetuneResponse:
        """
        Async method to cancel a running fine-tuning job

        Args:
            id (str): Fine-tune ID to cancel. A string that starts with `ft-`.

        Returns:
            FinetuneResponse: Object containing information about cancelled fine-tuning job.
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        response, _, _ = await requestor.arequest(
            options=TogetherRequest(
                method="POST",
                url=f"fine-tunes/{id}/cancel",
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)

        return FinetuneResponse(**response.data)

    async def list_events(self, id: str) -> FinetuneListEvents:
        """
        Async method to lists events of a fine-tune job

        Args:
            id (str): Fine-tune ID to list events for. A string that starts with `ft-`.

        Returns:
            FinetuneListEvents: Object containing list of fine-tune events
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        response, _, _ = await requestor.arequest(
            options=TogetherRequest(
                method="GET",
                url=f"fine-tunes/{id}/events",
            ),
            stream=False,
        )

        assert isinstance(response, TogetherResponse)

        return FinetuneListEvents(**response.data)

    async def download(
        self, id: str, *, output: str | None = None, checkpoint_step: int = -1
    ) -> str:
        """
        TODO: Implement async download method
        """

        raise NotImplementedError(
            "AsyncFineTuning.download not implemented. "
            "Please use FineTuning.download function instead."
        )

    async def get_model_limits(self, *, model: str) -> FinetuneTrainingLimits:
        """
        Requests training limits for a specific model

        Args:
            model_name (str): Name of the model to get limits for

        Returns:
            FinetuneTrainingLimits: Object containing training limits for the model
        """

        requestor = api_requestor.APIRequestor(
            client=self._client,
        )

        model_limits_response, _, _ = await requestor.arequest(
            options=TogetherRequest(
                method="GET",
                url="fine-tunes/models/limits",
                params={"model": model},
            ),
            stream=False,
        )

        model_limits = FinetuneTrainingLimits(**model_limits_response.data)

        return model_limits
