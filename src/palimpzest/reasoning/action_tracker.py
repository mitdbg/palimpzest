from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ActionState:
    this_step: dict
    remaining_questions: list[str]
    bad_attempts: int
    total_step: int

class ActionTracker:
    def __init__(self):
        self._listeners: dict[str, list[Callable]] = {}
        self.state = ActionState(
            this_step={"action": "answer", "answer": "", "references": [], "think": ""},
            remaining_questions=[],
            bad_attempts=0,
            total_step=0
        )

    def on(self, event: str, callback: Callable) -> None:
        """Add event listener"""
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)

    def emit(self, event: str, data: Any) -> None:
        """Emit event to all listeners"""
        if event in self._listeners:
            for callback in self._listeners[event]:
                callback(data)

    def track_action(self, new_state: dict) -> None:
        """Track new action state and emit event"""
        self.state = ActionState(**{**self.state.__dict__, **new_state})
        self.emit('action', self.state.this_step)

    def track_think(self, think: str, lang: str = None, params: dict = None) -> None:
        """Track thinking step and emit event"""
        self.state.this_step["think"] = think
        self.emit('action', self.state.this_step)