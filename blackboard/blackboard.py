from __future__ import annotations
import uuid
from schemas.timetable import Proposal, Assignment, RejectionRecord


class BlackBoard:
    def __init__(self):
        self._proposals: dict[str, Proposal] = {}
        self._assignments: list[Assignment] = []
        self._rejection_log: list[RejectionRecord] = []
        self._unscheduled_courses: list[str] = []


    # ── seeding ───────────────────────────────────────────────────────────────

    def seed(self, course_ids: list[str]) -> None:
        self._unscheduled_courses = list(course_ids)


    # ── reads ─────────────────────────────────────────────────────────────────

    def get_proposals(self) -> list[Proposal]:
        return list(self._proposals.values())

    def get_assignments(self) -> list[Assignment]:
        return list(self._assignments)

    def get_rejection_log(self) -> list[RejectionRecord]:
        return list(self._rejection_log)

    def get_unscheduled_courses(self) -> list[str]:
        return list(self._unscheduled_courses)

    def is_complete(self) -> bool:
        return not self._unscheduled_courses and not self._proposals


    # ── proposal lifecycle ────────────────────────────────────────────────────

    def add_proposal(self, proposal: Proposal) -> None:
        self._proposals[proposal.id] = proposal

    def update_proposal(self, proposal: Proposal) -> None:
        self._proposals[proposal.id] = proposal

    def confirm_proposal(self, proposal: Proposal, cycle: int) -> None:
        self._assignments.append(Assignment(
            course_id=proposal.course_id,
            room_id=proposal.room_id,
            lecturer_id=proposal.lecturer_id,
            timeslot=proposal.timeslot,
        ))
        self._unscheduled_courses = [
            c for c in self._unscheduled_courses if c != proposal.course_id
        ]
        del self._proposals[proposal.id]

    def abandon_proposal(self, proposal: Proposal, reason: str, cycle: int) -> None:
        self._rejection_log.append(RejectionRecord(
            course_id=proposal.course_id,
            reason=reason,
            cycle=cycle,
        ))
        self._unscheduled_courses = [
            c for c in self._unscheduled_courses if c != proposal.course_id
        ]
        del self._proposals[proposal.id]


    # ── proposal bootstrapping ────────────────────────────────────────────────

    def ensure_proposals_exist(self, cycle: int) -> None:
        """
        For any unscheduled course with no in-flight proposal, create one.
        One per cycle — steady drip, not a flood.
        """
        in_flight = {p.course_id for p in self._proposals.values()}
        for course_id in self._unscheduled_courses:
            if course_id not in in_flight:
                self.add_proposal(Proposal(id=str(uuid.uuid4()), course_id=course_id))
                return  # one per cycle


    # ── conflict detection — agents read these before proposing ──────────────

    def get_timeslot_conflicts(self, proposal: Proposal) -> list[Proposal]:
        """Other in-flight proposals already claiming this timeslot."""
        if proposal.timeslot is None:
            return []
        return [
            p for p in self._proposals.values()
            if p.id != proposal.id
            and p.timeslot == proposal.timeslot
        ]

    def get_room_conflicts(self, proposal: Proposal) -> list[Proposal]:
        """Other in-flight proposals claiming the same room at the same timeslot."""
        if proposal.timeslot is None or proposal.room_id is None:
            return []
        return [
            p for p in self._proposals.values()
            if p.id != proposal.id
            and p.room_id == proposal.room_id
            and p.timeslot == proposal.timeslot
        ]

    def get_lecturer_conflicts(self, proposal: Proposal) -> list[Proposal]:
        """Other in-flight proposals claiming the same lecturer at the same timeslot."""
        if proposal.timeslot is None or proposal.lecturer_id is None:
            return []
        return [
            p for p in self._proposals.values()
            if p.id != proposal.id
            and p.lecturer_id == proposal.lecturer_id
            and p.timeslot == proposal.timeslot
        ]