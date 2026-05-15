from dataclasses import dataclass

from blackboard.blackboard import BlackBoard

from schemas.course import Course
from schemas.room import Room
from schemas.lecturer import Lecturer
from schemas.policy import Policy

@dataclass
class Deps:
    board: BlackBoard            # mutable — agents read and write directly
    courses: list[Course]        # read-only reference data
    rooms: list[Room]            # read-only reference data
    lecturers: list[Lecturer]    # read-only reference data
    policy: Policy               # read-only reference data
    total_tokens: int = 0        # accumulated token usage across all agents