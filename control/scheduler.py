from __future__ import annotations
import json
from agents.base_agent import BaseAgent
from core.deps import Deps
from core.logger import logger

MAX_CYCLES = 50


class Scheduler:
    def __init__(self, agents: list[BaseAgent]):
        self._agents = agents

    async def run(self, deps: Deps) -> dict:
        board = deps.board
        cycle = 0

        while cycle < MAX_CYCLES:
            cycle += 1
            logger.info(f"[cycle {cycle}] unscheduled={board.get_unscheduled_courses()} "
                        f"in-flight={[p.id for p in board.get_proposals()]}")

            if board.is_complete():
                logger.info(f"[cycle {cycle}] board complete — done")
                break

            # each proposal reacts to its own state — scheduler stays blind to fields
            for proposal in list(board.get_proposals()):
                if proposal.is_exhausted:
                    logger.warning(f"[cycle {cycle}] abandoning {proposal.course_id}")
                    board.abandon_proposal(proposal, "Exceeded MAX_RETRIES", cycle)
                elif proposal.is_confirmed:
                    logger.info(f"[cycle {cycle}] confirming {proposal.course_id}")
                    board.confirm_proposal(proposal, cycle)
                elif proposal.is_rejected:
                    logger.info(f"[cycle {cycle}] resetting {proposal.course_id} "
                                f"— failed component: {proposal.failed_component}")
                    proposal.apply_rejection()
                    board.update_proposal(proposal)

            board.ensure_proposals_exist(cycle)

            # agents self-select — first competent agent activates
            for agent in self._agents:
                if proposal := agent.is_competent_for(board):
                    logger.info(f"[cycle {cycle}] {agent.name} self-activated "
                                f"for {proposal.course_id}")
                    await agent.run(proposal, deps)
                    break

        return self._produce_output(deps, cycle)

    def _produce_output(self, deps: Deps, cycle: int) -> dict:
        board = deps.board
        courses   = {c.id: c for c in deps.courses}
        rooms     = {r.id: r for r in deps.rooms}
        lecturers = {l.id: l for l in deps.lecturers}
        rejection_index = {r.course_id: r.reason for r in board.get_rejection_log()}

        assignments = [
            {
                "course_id":      a.course_id,
                "course_name":    courses[a.course_id].name,
                "room_id":        a.room_id,
                "room_name":      rooms[a.room_id].name,
                "lecturer_id":    a.lecturer_id,
                "lecturer_name":  lecturers[a.lecturer_id].name,
                "day":            a.timeslot.day,
                "start_hour":     a.timeslot.start_hour,
                "end_hour":       a.timeslot.end_hour,
            }
            for a in board.get_assignments()
        ]

        unresolved = [
            {
                "course_id": cid,
                "reason": rejection_index.get(cid, "Unknown"),
            }
            for cid in board.get_unscheduled_courses()
        ]

        return {
            "generated":    len(unresolved) == 0,
            "total_cycles": cycle,
            "total_tokens": deps.total_tokens,
            "assignments":  assignments,
            "unresolved":   unresolved,
        }