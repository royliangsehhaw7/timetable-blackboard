from __future__ import annotations
from abc import ABC, abstractmethod
from pydantic_ai import Agent
from blackboard.blackboard import BlackBoard
from schemas.timetable import Proposal


class BaseAgent(ABC):
    def __init__(self, name: str, agent: Agent | None):
        self._name = name
        self._agent = agent

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def is_competent_for(self, board: BlackBoard) -> Proposal | None:
        """
        Inspect the board and return the proposal this agent should work on,
        or None if no current work matches this agent's competency.
        This is the sole activation mechanism — no external dispatch.
        """
        ...

    @abstractmethod
    def get_instruction(self) -> str: 
        ...

    def get_failure_prompt(self, failure_context: str) -> str:
        return (
            f"CORRECTION REQUIRED\n"
            f"Your previous result was rejected: {failure_context}\n"
            f"Return a different result that avoids this problem."
        )