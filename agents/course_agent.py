from __future__ import annotations
import json
from pydantic_ai import Agent

from agents.base_agent import BaseAgent
from blackboard.blackboard import BlackBoard
from core.deps import Deps
from schemas.timetable import Proposal


class CourseAgent(BaseAgent):
    def __init__(self, name: str, agent: Agent):
        super().__init__(name, agent)

    def is_competent_for(self, board: BlackBoard) -> Proposal | None:
        return next(
            (p for p in board.get_proposals() if p.timeslot is None), None,
        )

    def get_instruction(self) -> str:
        return (f"""
                You are the CourseAgent in a timetable scheduling system.
                Your sole job is to propose the best timeslot for a given course.
                
                Rules:\n"
                - Timeslot must be on a valid school day, within school hours, not during lunch
                - Avoid timeslots already taken by confirmed assignments
                - Avoid timeslots already claimed by other in-progress proposals
                - Spread courses across the week rather than clustering them
                
                Call log_decision to explain your reasoning. Return the proposal with timeslot set.
            """
        )

    def get_prompt(self, proposal: Proposal, deps: Deps) -> str:
        course      = next(c for c in deps.courses if c.id == proposal.course_id)
        confirmed   = deps.board.get_assignments()
        in_conflict = deps.board.get_timeslot_conflicts(proposal)

        prompt = (f"""
                Course:\n{json.dumps(course.model_dump(), indent=2)}
                School policy:\n{json.dumps(deps.policy.model_dump(), indent=2)}

                Confirmed assignments (timeslots taken):
                {json.dumps([a.model_dump() for a in confirmed], indent=2)}
                Timeslots already claimed by other in-progress proposals (avoid these too):
                {json.dumps([p.timeslot.model_dump() for p in in_conflict], indent=2)}
            """
        )

        if proposal.failure_context:
            prompt += f"\n\n{self.get_failure_prompt(proposal.failure_context)}"
        
        return prompt

    async def run(self, proposal: Proposal, deps: Deps) -> None:
        result  = await self._agent.run(
            self.get_prompt(proposal, deps),
            deps=deps,
            instructions=self.get_instruction(),
        )
        
        deps.board.update_proposal(result.output)