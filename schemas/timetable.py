from pydantic import BaseModel
from .timeslot import TimeSlot

class Proposal(BaseModel):
    id: str
    course_id: str
    timeslot: TimeSlot | None = None
    room_id: str | None = None
    lecturer_id: str | None = None
    policy_approved: bool | None = False
    policy_reason: str | None = None
    retry_count: int = 0

    failed_component: str | None = None   # "timeslot" | "room" | "lecturer" — set by PolicyAgent on rejection
    failure_context: str | None = None    # populated by scheduler from policy_reason after rejection


class Assignment(BaseModel):
    course_id: str
    room_id: str
    lecturer_id: str
    timeslot: TimeSlot


class RejectionRecord(BaseModel):
    course_id: str
    reason: str
    cycle: int
