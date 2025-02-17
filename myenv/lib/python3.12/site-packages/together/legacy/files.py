from __future__ import annotations

import json
import warnings
from typing import Any, Dict, List

import together
from together.legacy.base import API_KEY_WARNING, deprecated
from together.utils.files import check_file as check_json


class Files:
    @classmethod
    @deprecated  # type: ignore
    def list(
        cls,
    ) -> Dict[str, Any]:
        """Legacy file list function."""

        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        client = together.Together(api_key=api_key)

        return client.files.list().model_dump(exclude_none=True)

    @classmethod
    def check(self, file: str) -> Dict[str, object]:
        return check_json(file)

    @classmethod
    @deprecated  # type: ignore
    def upload(
        cls,
        file: str,
        check: bool = True,
    ) -> Dict[str, Any]:
        """Legacy file upload function."""

        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        if check:
            report_dict = check_json(file)
            if not report_dict["is_check_passed"]:
                raise together.error.FileTypeError(
                    f"Invalid file supplied. Failed to upload.\nReport:\n {report_dict}"
                )

        client = together.Together(api_key=api_key)

        # disabling the check, because it was run previously
        response = client.files.upload(file=file, check=False).model_dump(
            exclude_none=True
        )

        if check:
            response["report_dict"] = report_dict

        return response

    @classmethod
    @deprecated  # type: ignore
    def delete(
        cls,
        file_id: str,
    ) -> Dict[str, Any]:
        """Legacy file delete function."""

        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        client = together.Together(api_key=api_key)

        return client.files.delete(id=file_id).model_dump(exclude_none=True)

    @classmethod
    @deprecated  # type: ignore
    def retrieve(
        cls,
        file_id: str,
    ) -> Dict[str, Any]:
        """Legacy file retrieve function."""

        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        client = together.Together(api_key=api_key)

        return client.files.retrieve(id=file_id).model_dump(exclude_none=True)

    @classmethod
    @deprecated  # type: ignore
    def retrieve_content(
        cls,
        file_id: str,
        output: str | None = None,
    ) -> Dict[str, Any]:
        """Legacy file retrieve content function."""

        api_key = None
        if together.api_key:
            warnings.warn(API_KEY_WARNING)
            api_key = together.api_key

        client = together.Together(api_key=api_key)

        return client.files.retrieve_content(id=file_id, output=output).dict(
            exclude_none=True
        )

    @classmethod
    @deprecated  # type: ignore
    def save_jsonl(
        self, data: Dict[str, str], output_path: str, append: bool = False
    ) -> None:
        """
        Write list of objects to a JSON lines file.
        """
        mode = "a+" if append else "w"
        with open(output_path, mode, encoding="utf-8") as f:
            for line in data:
                json_record = json.dumps(line, ensure_ascii=False)
                f.write(json_record + "\n")
        print("Wrote {} records to {}".format(len(data), output_path))

    @classmethod
    @deprecated  # type: ignore
    def load_jsonl(self, input_path: str) -> List[Dict[str, str]]:
        """
        Read list of objects from a JSON lines file.
        """
        data = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line.rstrip("\n|\r")))
        print("Loaded {} records from {}".format(len(data), input_path))
        return data
