from __future__ import annotations

import json
import os
from pathlib import Path
from traceback import format_exc
from typing import Any, Dict

from pyarrow import ArrowInvalid, parquet

from together.constants import (
    MAX_FILE_SIZE_GB,
    MIN_SAMPLES,
    NUM_BYTES_IN_GB,
    PARQUET_EXPECTED_COLUMNS,
)


def check_file(
    file: Path | str,
) -> Dict[str, Any]:
    if not isinstance(file, Path):
        file = Path(file)

    report_dict = {
        "is_check_passed": True,
        "message": "Checks passed",
        "found": None,
        "file_size": None,
        "utf8": None,
        "line_type": None,
        "text_field": None,
        "key_value": None,
        "min_samples": None,
        "num_samples": None,
        "load_json": None,
    }

    if not file.is_file():
        report_dict["found"] = False
        report_dict["is_check_passed"] = False
        return report_dict
    else:
        report_dict["found"] = True

    file_size = os.stat(file.as_posix()).st_size

    if file_size > MAX_FILE_SIZE_GB * NUM_BYTES_IN_GB:
        report_dict["message"] = (
            f"Maximum supported file size is {MAX_FILE_SIZE_GB} GB. Found file with size of {round(file_size / NUM_BYTES_IN_GB ,3)} GB."
        )
        report_dict["is_check_passed"] = False
    elif file_size == 0:
        report_dict["message"] = "File is empty"
        report_dict["file_size"] = 0
        report_dict["is_check_passed"] = False
        return report_dict
    else:
        report_dict["file_size"] = file_size

    if file.suffix == ".jsonl":
        report_dict["filetype"] = "jsonl"
        data_report_dict = _check_jsonl(file)
    elif file.suffix == ".parquet":
        report_dict["filetype"] = "parquet"
        data_report_dict = _check_parquet(file)
    else:
        report_dict["filetype"] = (
            f"Unknown extension of file {file}. "
            "Only files with extensions .jsonl and .parquet are supported."
        )
        report_dict["is_check_passed"] = False

    report_dict.update(data_report_dict)
    return report_dict


def _check_jsonl(file: Path) -> Dict[str, Any]:
    report_dict: Dict[str, Any] = {}
    # Check that the file is UTF-8 encoded. If not report where the error occurs.
    try:
        with file.open(encoding="utf-8") as f:
            f.read()
        report_dict["utf8"] = True
    except UnicodeDecodeError as e:
        report_dict["utf8"] = False
        report_dict["message"] = f"File is not UTF-8 encoded. Error raised: {e}."
        report_dict["is_check_passed"] = False
        return report_dict

    with file.open() as f:
        # idx must be instantiated so decode errors (e.g. file is a tar) or empty files are caught
        idx = -1
        try:
            for idx, line in enumerate(f):
                json_line = json.loads(line)  # each line in jsonlines should be a json

                if not isinstance(json_line, dict):
                    report_dict["line_type"] = False
                    report_dict["message"] = (
                        f"Error parsing file. Invalid format on line {idx + 1} of the input file. "
                        'Example of valid json: {"text": "my sample string"}. '
                    )

                    report_dict["is_check_passed"] = False

                if "text" not in json_line.keys():
                    report_dict["text_field"] = False
                    report_dict["message"] = (
                        f"Missing 'text' field was found on line {idx + 1} of the the input file. "
                        "Expected format: {'text': 'my sample string'}. "
                    )
                    report_dict["is_check_passed"] = False
                else:
                    # check to make sure the value of the "text" key is a string
                    if not isinstance(json_line["text"], str):
                        report_dict["key_value"] = False
                        report_dict["message"] = (
                            f'Invalid value type for "text" key on line {idx + 1}. '
                            f'Expected string. Found {type(json_line["text"])}.'
                        )

                        report_dict["is_check_passed"] = False

            # make sure this is outside the for idx, line in enumerate(f): for loop
            if idx + 1 < MIN_SAMPLES:
                report_dict["min_samples"] = False
                report_dict["message"] = (
                    f"Processing {file} resulted in only {idx + 1} samples. "
                    f"Our minimum is {MIN_SAMPLES} samples. "
                )
                report_dict["is_check_passed"] = False
            else:
                report_dict["num_samples"] = idx + 1
                report_dict["min_samples"] = True

            report_dict["load_json"] = True

        except ValueError:
            report_dict["load_json"] = False
            if idx < 0:
                report_dict["message"] = (
                    "Unable to decode file. "
                    "File may be empty or in an unsupported format. "
                )
            else:
                report_dict["message"] = (
                    f"Error parsing json payload. Unexpected format on line {idx + 1}."
                )
            report_dict["is_check_passed"] = False

    if "text_field" not in report_dict:
        report_dict["text_field"] = True
    if "line_type" not in report_dict:
        report_dict["line_type"] = True
    if "key_value" not in report_dict:
        report_dict["key_value"] = True
    return report_dict


def _check_parquet(file: Path) -> Dict[str, Any]:
    report_dict: Dict[str, Any] = {}

    try:
        table = parquet.read_table(str(file), memory_map=True)
    except ArrowInvalid:
        report_dict["load_parquet"] = (
            f"An exception has occurred when loading the Parquet file {file}. Please check the file for corruption. "
            f"Exception trace:\n{format_exc()}"
        )
        report_dict["is_check_passed"] = False
        return report_dict

    column_names = table.schema.names
    if "input_ids" not in column_names:
        report_dict["load_parquet"] = (
            f"Parquet file {file} does not contain the `input_ids` column."
        )
        report_dict["is_check_passed"] = False
        return report_dict

    for column_name in column_names:
        if column_name not in PARQUET_EXPECTED_COLUMNS:
            report_dict["load_parquet"] = (
                f"Parquet file {file} contains an unexpected column {column_name}. "
                f"Only columns {PARQUET_EXPECTED_COLUMNS} are supported."
            )
            report_dict["is_check_passed"] = False
            return report_dict

    num_samples = len(table)
    if num_samples < MIN_SAMPLES:
        report_dict["min_samples"] = (
            f"Processing {file} resulted in only {num_samples} samples. "
            f"Our minimum is {MIN_SAMPLES} samples. "
        )
        report_dict["is_check_passed"] = False
        return report_dict
    else:
        report_dict["num_samples"] = num_samples

    report_dict["is_check_passed"] = True

    return report_dict
