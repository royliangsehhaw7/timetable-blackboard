from pydantic_ai import Agent


from core.deps import Deps
from agents.base_agent import BaseAgent

from blackboard.blackboard import BlackBoard
from schemas.timetable import Proposal


class PolicyAgent(BaseAgent):
    def __init__(self, name: str, agent: Agent = None):
        super().__init__(name, agent) 
    def is_competent_for(self, board: BlackBoard) -> Proposal | None:
        return next(
            (p for p in board.get_proposals()
                # if p.lecturer_id is not None and p.policy_approved is None),
                if p.room_id is not None and p.policy_approved is None),
            None,
        )

    def get_instruction(self) -> str:
        return ""  # no LLM

    def check(self, proposal: Proposal, deps: Deps) -> tuple[bool, str | None, str | None]:
        """
        Returns (approved, reason, failed_component).
        Checks policy rules first, then inter-proposal conflicts.
        failed_component is one of "timeslot", "room", "lecturer", or None if approved.
        """
        policy   = deps.policy
        course   = next(c for c in deps.courses   if c.id == proposal.course_id)
        room     = next(r for r in deps.rooms      if r.id == proposal.room_id)
        # lecturer = next(l for l in deps.lecturers  if l.id == proposal.lecturer_id)
        slot     = proposal.timeslot

        
        # ── policy checks ─────────────────────────────────────────────────────
        if slot.day not in policy.school_days:
            return False, f"{slot.day} is not a valid school day", "timeslot"

        if slot.start_hour < policy.school_start_hour or slot.end_hour > policy.school_end_hour:
            return False, "Timeslot falls outside school hours", "timeslot"

        if slot.start_hour < policy.lunch_end_hour and slot.end_hour > policy.lunch_start_hour:
            return False, "Timeslot overlaps lunch break", "timeslot"

        if course.requires_lab and room.room_type != "lab":
            return False, f"Course requires a lab but {room.name} is a {room.room_type}", "room"

        # if proposal.course_id not in lecturer.courses_taught:
        #     return False, f"{lecturer.name} is not qualified to teach this course", "lecturer"

        if any(s.day == slot.day and s.start_hour == slot.start_hour
               for s in lecturer.unavailable_slots):
            return False, f"{lecturer.name} is unavailable at this timeslot", "lecturer"

        
        # ── inter-proposal conflict checks ────────────────────────────────────
        room_conflicts = deps.board.get_room_conflicts(proposal)
        if room_conflicts:
            other = room_conflicts[0].course_id
            return False, f"{room.name} already claimed by in-flight proposal for {other}", "room"

        # lecturer_conflicts = deps.board.get_lecturer_conflicts(proposal)
        # if lecturer_conflicts:
        #     other = lecturer_conflicts[0].course_id
        #     return False, f"{lecturer.name} already claimed by in-flight proposal for {other}", "lecturer"

        return True, None, None

    async def run(self, proposal: Proposal, deps: Deps) -> None:
        approved, reason, failed_component = self.check(proposal, deps)
        proposal.policy_approved  = approved
        proposal.policy_reason    = reason
        proposal.failed_component = failed_component
        deps.board.update_proposal(proposal)