import asyncio
import logging

from pydantic_ai import Agent

from agents.course_agent import CourseAgent
from agents.room_agent import RoomAgent
from agents.policy_agent import PolicyAgent

from core.deps import Deps
from core.data_loader import load_data
from core.llm_factory import LLMFactory
from blackboard.blackboard import BlackBoard

from schemas.timetable import Proposal


from control.scheduler import Scheduler
from tools.agent_logger import log_decision


async def main():
    # =====
    blackboard = BlackBoard()
    courses, rooms, lecturers, policy = load_data()
    blackboard.seed([c.id for c in courses])
    #
    deps = Deps(
        board = blackboard,
        courses = courses,
        rooms = rooms,
        lecturers = lecturers,
        policy = policy,
        total_tokens = 0
    )

    # ===== create all agents
    factory = LLMFactory("openrouter")
    model = factory.get_model(model="nvidia/nemotron-3-super-120b-a12b:free")

    cAgent = Agent(
        model=model,
        output_type=Proposal,
        tools = [log_decision]
    )
    courseAgent = CourseAgent(name="CourseAgent", agent=cAgent)
    rAgent = Agent(
        model = model,
        output_type = Proposal,
        tools=[log_decision]
    )
    roomAgent = RoomAgent(name="RoomAgent", agent=rAgent)
    pAgent = Agent(
        model = model,
        output_type=Proposal,
        tools=[log_decision]
    )
    policyAgent = PolicyAgent(name="PolicyAgent", agent=pAgent)

    # =====
    scheduler = Scheduler([courseAgent,roomAgent,policyAgent])
    await scheduler.run(deps)


if __name__ == "__main__":
    asyncio.run(main())
