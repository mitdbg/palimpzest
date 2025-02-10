import hashlib
import json

from palimpzest.constants import MAX_ID_CHARS


def hash_for_id(id_str: str, max_chars: int = MAX_ID_CHARS) -> str:
    return hashlib.sha256(id_str.encode("utf-8")).hexdigest()[:max_chars]


def hash_for_serialized_dict(dict_obj: dict) -> str:
    return hash_for_id(json.dumps(dict_obj, sort_keys=True))
