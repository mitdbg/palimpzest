from dataclasses import dataclass
from enum import Enum


@dataclass
class Reference:
    exact_quote: str
    url: str
    title: str = ""
    date_time: str = ""

class ActionType(Enum):
    ANSWER = "answer"
    REFLECT = "reflect"
    SEARCH = "search"
    VISIT = "visit"

@dataclass
class ExecuteAction:
    action: ActionType
    think: str = ""
    answer: str = ""
    reflections: dict = None
    is_final: bool = False
    questions_to_answer: list[str] = None 
