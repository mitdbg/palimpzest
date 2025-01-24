import hashlib

from palimpzest.constants import MAX_ID_CHARS


def hash_for_id(id_str: str, max_chars: int = MAX_ID_CHARS) -> str:
    return hashlib.sha256(id_str.encode("utf-8")).hexdigest()[:max_chars]


def hash_for_temp_schema(id_str:str) ->str:
    return hash_for_id(id_str)
