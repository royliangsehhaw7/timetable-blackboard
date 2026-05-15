from __future__ import annotations
import json
from pydantic_ai import Agent

from agents.base_agent import BaseAgent
from blackboard.blackboard import BlackBoard
from core.deps import Deps
from schemas.timetable import Proposal


class RoomAgent(BaseAgent):
    def __init__(self, name: str, agent: Agent):
        super().__init__(name, agent)

    def is_competent_for(self, board: BlackBoard) -> Proposal | None:
        return next(
            (p for p in board.get_proposals()
             if p.timeslot is not None and p.room_id is None),
            None,
        )

    def get_instruction(self) -> str:
        return (f"""
                You are the RoomAgent in a timetable scheduling system.
                Your sole job is to assign the most suitable room for a course proposal.
                Rules:
                - Match room type to course requirement (lab → lab room, non-lab → classroom)
                - Never assign a room already confirmed at this timeslot
                - Never assign a room already claimed by another in-progress proposal at this timeslot
                - If multiple suitable rooms are free, prefer the best fit
                
                Call log_decision to explain your reasoning. Return the proposal with room_id set.
            """
        )

    def get_prompt(self, proposal: Proposal, deps: Deps) -> str:
        course      = next(c for c in deps.courses if c.id == proposal.course_id)
        confirmed   = [
            a for a in deps.board.get_assignments()
            if a.timeslot == proposal.timeslot
        ]
        in_conflict = deps.board.get_room_conflicts(proposal)

        prompt = (f"""
                Current proposal:\n{json.dumps(proposal.model_dump(), indent=2)}
                Course:\n{json.dumps(course.model_dump(), indent=2)}
                All rooms:\n{json.dumps([r.model_dump() for r in deps.rooms], indent=2)}
                Rooms confirmed at this timeslot:
                {json.dumps([a.model_dump() for a in confirmed], indent=2)}
                Rooms already claimed by other in-progress proposals at this timeslot:
                {json.dumps([p.room_id for p in in_conflict], indent=2)}
            """
        )
        if proposal.failure_context:
            prompt += f"\n\n{self.get_failure_prompt(proposal.failure_context)}"

        return prompt

    async def run(self, proposal: Proposal, deps: Deps) -> None:
        result = await self._agent.run(
            self.get_prompt(proposal, deps),
            deps=deps,
            instructions=self.get_instruction(),
        )

        deps.board.update_proposal(result.output)