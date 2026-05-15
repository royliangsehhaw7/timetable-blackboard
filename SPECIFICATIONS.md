# Multi-Agent Timetable Generation System
## Specification v3.0 — Blackboard / Competency-Based Role Pattern (Revised)

---

## 1. Purpose

A learning-oriented multi-agent system that generates a valid weekly school timetable. The primary
goal is to understand and implement the **Blackboard Pattern with Competency-Based Roles** in a
grounded, practical context using pydantic-ai's deps, agents, and tools features properly.

**What is different from v2.0:**

v2.0 had three structural problems that undermined the blackboard pattern:

1. **The board did domain reasoning.** `get_timeslot_conflicts()`, `get_room_conflicts()`, and
   `get_lecturer_conflicts()` are domain logic — conflict reasoning about scheduling. A blackboard
   is a dumb data store. Domain reasoning belongs in the agents that own it.

2. **`Proposal` was a state machine, not a schema.** It contained `apply_rejection()`,
   `_reset_timeslot/room/lecturer`, and a `_RESET_STRATEGY` dispatch dict. A Pydantic `BaseModel`
   describes data shape. Lifecycle and mutation logic belongs in the component that manages
   lifecycle — the Scheduler.

3. **LLM agents received pre-filtered context instead of reading the board.** True blackboard
   agents walk up to the board and read what they need. The board should expose raw state; agents
   decide what is relevant to their competency.

v3.0 corrects all three. The blackboard is a pure data store. `Proposal` is a pure schema.
The Scheduler owns all lifecycle transitions. Agents read raw board state and apply their own
domain reasoning.

---

## 2. Architecture

### 2.1 Pattern

This system implements the **Blackboard Pattern with Competency-Based Roles**.

A shared `BlackBoard` object holds all system state — proposals in progress, confirmed assignments,
rejection history, and unscheduled course ids. Every agent reads directly from the blackboard and
writes directly back to it. No coordinator mediates these reads or writes. The blackboard does not
reason about its own contents.

Each agent defines its own **competency guard** — a method that inspects the blackboard and returns
the proposal it should work on, or `None` if no work is currently relevant. The scheduler loop
calls every agent's guard every cycle. Whichever agent finds relevant work activates itself.

The sequential constraint of the scheduling pipeline — timeslot → room → lecturer → policy — is
enforced by each agent's guard condition, not by external dispatch:

- `CourseAgent` activates when a proposal exists with no timeslot
- `RoomAgent` activates when a proposal has a timeslot but no room
- `LecturerAgent` activates when a proposal has a room but no lecturer
- `PolicyAgent` activates when a proposal is fully assembled but unvalidated

No agent is told to go. Each agent decides for itself whether it is relevant to the current state.

### 2.2 The Blackboard

The `BlackBoard` is the single source of truth and a **pure data store**. It holds state. It does
not compute conflicts, filter proposals by agent type, or make any domain decisions. All agents
read raw state from it and apply their own domain logic locally.

### 2.3 Agents

There are four agents. There is no orchestrator.

| Agent | Competency — activates when |
|---|---|
| `CourseAgent` | A proposal exists on the board with no timeslot set |
| `RoomAgent` | A proposal exists with a timeslot but no room_id |
| `LecturerAgent` | A proposal exists with a room_id but no lecturer_id |
| `PolicyAgent` | A proposal exists that is fully assembled but policy not yet evaluated |

`CourseAgent`, `RoomAgent`, and `LecturerAgent` are LLM-powered. `PolicyAgent` is pure Python —
policy checking is fully deterministic.

Each agent reads raw proposals from the board, applies its own conflict reasoning, constructs its
own context, calls the LLM (or runs Python), and writes back to the board directly.

### 2.4 Flow

```
LOOP:
    board state snapshot logged

    if board.is_complete() → done

    for each in-flight proposal:
        if retry_count >= MAX_RETRIES → scheduler abandons it
        if policy_approved is True   → scheduler confirms it
        if policy_approved is False  → scheduler resets it (see 2.5)

    board.ensure_proposals_exist()   # one new proposal per cycle if needed

    for each agent in [CourseAgent, RoomAgent, LecturerAgent, PolicyAgent]:
        proposal = agent.is_competent_for(board)
        if proposal:
            await agent.run(proposal, deps)   # agent reads board, writes back
            break                             # one agent activates per cycle
```

### 2.5 Failure Recovery

When `PolicyAgent` rejects a proposal it sets `policy_approved=False`, `policy_reason` with a
description, and `failed_component` with one of `"timeslot"`, `"room"`, `"lecturer"`.

The **Scheduler** reads these fields and resets the proposal accordingly:

```python
if proposal.failed_component == "timeslot":
    proposal.timeslot    = None
    proposal.room_id     = None
    proposal.lecturer_id = None
elif proposal.failed_component == "room":
    proposal.room_id     = None
    proposal.lecturer_id = None
elif proposal.failed_component == "lecturer":
    proposal.lecturer_id = None

proposal.failure_context  = proposal.policy_reason
proposal.retry_count     += 1
proposal.policy_approved  = None
proposal.policy_reason    = None
proposal.failed_component = None

board.update_proposal(proposal)
```

This reset logic lives entirely in the Scheduler. `Proposal` does not mutate itself.

On the next cycle, the agent whose guard matches the reset field self-activates and receives
`failure_context` from the proposal it finds on the board.

### 2.6 What no agent does

No agent tells another agent what to do. No agent reads the full board to make routing decisions.
No agent knows the full pipeline. Each agent knows only its own competency condition and its own
domain work.

---

## 3. Project Structure

```
timetable/
├── agents/
│   ├── base_agent.py
│   ├── course_agent.py
│   ├── room_agent.py
│   ├── lecturer_agent.py
│   └── policy_agent.py
├── blackboard/
│   └── blackboard.py
├── control/
│   └── scheduler.py
├── core/
│   ├── data_loader.py
│   ├── deps.py
│   ├── logger.py
│   └── llm_factory.py
├── schemas/
│   ├── timeslot.py
│   ├── course.py
│   ├── room.py
│   ├── lecturer.py
│   ├── policy.py
│   └── timetable.py
├── tools/
│   └── agent_logger.py
├── data/
│   ├── courses.json
│   ├── rooms.json
│   ├── lecturers.json
│   └── policy.json
└── main.py
```

No structural changes from v2.0. All file locations are identical.

---

## 4. Schemas

**Location**: `schemas/`

All files are plain Pydantic `BaseModel` classes. No methods, no logic, no imports from any other
project module.

**`schemas/timeslot.py`**

```python
class TimeSlot(BaseModel):
    day: str          # e.g. "Monday"
    start_hour: int   # e.g. 10
    end_hour: int     # e.g. 11
```

**`schemas/course.py`**

```python
class Course(BaseModel):
    id: str
    name: str
    requires_lab: bool
```

**`schemas/room.py`**

```python
class Room(BaseModel):
    id: str
    name: str
    room_type: str    # "lab" or "classroom"
```

**`schemas/lecturer.py`**

```python
class Lecturer(BaseModel):
    id: str
    name: str
    courses_taught: list[str]
    unavailable_slots: list[TimeSlot]
```

**`schemas/policy.py`**

```python
class Policy(BaseModel):
    school_days: list[str]
    school_start_hour: int
    school_end_hour: int
    lunch_start_hour: int
    lunch_end_hour: int
```

**`schemas/timetable.py`**

```python
MAX_RETRIES = 5

class Proposal(BaseModel):
    id: str
    course_id: str
    timeslot: TimeSlot | None = None
    room_id: str | None = None
    lecturer_id: str | None = None
    policy_approved: bool | None = None
    policy_reason: str | None = None
    failed_component: str | None = None   # "timeslot" | "room" | "lecturer"
    failure_context: str | None = None
    retry_count: int = 0


class Assignment(BaseModel):
    course_id: str
    room_id: str
    lecturer_id: str
    timeslot: TimeSlot
    cycle: int


class RejectionRecord(BaseModel):
    course_id: str
    reason: str
    cycle: int
```

**Key change from v2.0:** `Proposal` is a pure data schema. All state predicates (`is_confirmed`,
`is_rejected`, `is_exhausted`) and all mutation methods (`apply_rejection`, `_reset_*`,
`_RESET_STRATEGY`) are removed. The Scheduler reads `policy_approved`, `retry_count`, and
`failed_component` directly and handles all transitions itself.

**Invariant**: `Proposal` has no methods beyond what Pydantic provides. No mutation, no self-repair,
no strategy dispatch.

**Verification**: import each schema in a scratch script and instantiate one instance per model
with dummy data. All fields should round-trip through `model_dump()` and `model_validate()` without
error. Confirm `failed_component` and `failure_context` default to `None`.

---

## 5. The BlackBoard

**Location**: `blackboard/blackboard.py`

The blackboard is the shared knowledge base and a **pure data store**. It holds raw state. It
does not compute conflicts, filter by agent, or reason about its contents in any way. Agents read
raw state and apply their own domain logic.

### 5.1 What it holds

```python
class BlackBoard:
    def __init__(self):
        self._proposals: dict[str, Proposal] = {}
        self._assignments: list[Assignment] = []
        self._rejection_log: list[RejectionRecord] = []
        self._unscheduled_courses: list[str] = []

    def is_complete(self) -> bool:
        return not self._unscheduled_courses and not self._proposals

    def ensure_proposals_exist(self, cycle: int) -> None:
        in_flight = {p.course_id for p in self._proposals.values()}
        for course_id in self._unscheduled_courses:
            if course_id not in in_flight:
                self.add_proposal(Proposal(id=str(uuid.uuid4()), course_id=course_id))
                return
```

**Key change from v2.0:** `get_timeslot_conflicts()`, `get_room_conflicts()`, and
`get_lecturer_conflicts()` are removed entirely. These were domain logic — conflict reasoning about
scheduling — which does not belong in a data store. Each agent reads raw proposals from the board
and computes conflicts itself as part of its domain work.

### 5.2 Methods

#### Seeding

| Method | Signature | What it does |
|---|---|---|
| `seed` | `(course_ids: list[str]) -> None` | Populates `_unscheduled_courses` at startup. |

#### Proposal lifecycle

| Method | Signature | What it does |
|---|---|---|
| `add_proposal` | `(proposal: Proposal) -> None` | Adds a new in-progress proposal. |
| `update_proposal` | `(proposal: Proposal) -> None` | Replaces the existing proposal with the same id. |
| `confirm_proposal` | `(proposal: Proposal, cycle: int) -> None` | Constructs an `Assignment`, removes course from unscheduled, removes proposal. |
| `abandon_proposal` | `(proposal: Proposal, reason: str, cycle: int) -> None` | Appends a `RejectionRecord`, removes course from unscheduled, removes proposal. |

#### Read methods

| Method | Signature | What it returns |
|---|---|---|
| `get_proposals` | `() -> list[Proposal]` | All in-progress proposals. Raw — no filtering. |
| `get_assignments` | `() -> list[Assignment]` | All confirmed assignments. |
| `get_rejection_log` | `() -> list[RejectionRecord]` | Full rejection history. |
| `get_unscheduled_courses` | `() -> list[str]` | Course ids not yet confirmed or abandoned. |

No domain logic lives in the blackboard. No routing. No conflict detection. Just reads and writes.

**Verification**: seed with three course ids. Add two proposals. Update one. Confirm one and verify
it appears in assignments and is removed from proposals and unscheduled. Abandon one and verify
rejection log is updated. Print all collections after each operation.

---

## 6. Dependencies — Deps and RunContext

### 6.1 What Deps holds

```python
@dataclass
class Deps:
    board: BlackBoard            # mutable — agents read and write directly
    courses: list[Course]        # read-only reference data
    rooms: list[Room]            # read-only reference data
    lecturers: list[Lecturer]    # read-only reference data
    policy: Policy               # read-only reference data
    total_tokens: int = 0
```

### 6.2 How Deps flows at runtime

One `Deps` instance is created before the scheduler loop and passed into every agent's `run()`
call. pydantic-ai injects it into `RunContext` so the `log_decision` tool can accumulate token
usage into `deps.total_tokens`.

---

## 7. Tools

### 7.1 Design decision

There is one tool: `log_decision`. It is registered on all LLM-powered agents. A tool is something
the LLM calls mid-reasoning to perform an action with a side effect. `log_decision` qualifies: the
LLM triggers it deliberately to narrate its reasoning, and it produces structured log output and
accumulates token counts as side effects.

Blackboard writes happen inside each agent's `run()` method or inside the Scheduler. They are not
tools because they are deterministic Python operations, not LLM-triggered actions.

### 7.2 The tool

**Location**: `tools/agent_logger.py`

```python
import logging
from pydantic_ai import RunContext
from core.deps import Deps

logger = logging.getLogger(__name__)

async def log_decision(ctx: RunContext[Deps], message: str) -> str:
    """Call this once to explain your reasoning before returning your result.
    Describe what you found, what you decided, and why."""
    usage = ctx.usage
    ctx.deps.total_tokens += usage.total_tokens or 0
    logger.info(
        f"[decision] {message} | "
        f"tokens: request={usage.request_tokens} "
        f"response={usage.response_tokens} "
        f"total={usage.total_tokens}"
    )
    return "logged"
```

### 7.3 Tool registration per agent

| Tool | CourseAgent | RoomAgent | LecturerAgent | PolicyAgent |
|---|:---:|:---:|:---:|:---:|
| `log_decision` | ✓ | ✓ | ✓ | — |

---

## 8. Agents

### 8.1 Design principles

Each agent has one domain responsibility and one competency guard. The guard is pure Python — no
LLM involved. The domain work is LLM-powered (except PolicyAgent).

Each agent has three methods:

- `is_competent_for(board)` — inspects the blackboard, returns the first proposal this agent
  should work on, or `None`. Called by the scheduler every cycle.
- `get_instruction()` — returns the agent's stable identity and rules. Passed as the system turn.
- `get_prompt(proposal, deps)` — reads raw state from the board, applies local conflict reasoning,
  and constructs the full situational context for this activation. Passed as the user turn.

Each agent's `run()` calls the LLM (or runs Python), receives a structured `Proposal` back, and
writes it to the blackboard via `deps.board.update_proposal()`. The agent writes to the board
itself — the Scheduler does not do this on its behalf.

**Key change from v2.0:** `get_prompt()` no longer receives pre-filtered conflict lists. Each
agent reads `deps.board.get_proposals()` and `deps.board.get_assignments()` directly and computes
its own conflict view. The board's conflict methods are gone; that reasoning now lives here.

### 8.2 BaseAgent

**Location**: `agents/base_agent.py`

```python
from abc import ABC, abstractmethod
from pydantic_ai import Agent
from blackboard.blackboard import BlackBoard
from schemas.timetable import Proposal

class BaseAgent(ABC):
    def __init__(self, name: str, agent: Agent):
        self._name = name
        self._agent = agent

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def is_competent_for(self, board: BlackBoard) -> Proposal | None: ...

    @abstractmethod
    def get_instruction(self) -> str: ...

    def get_failure_prompt(self, failure_context: str) -> str:
        return f"""
            CORRECTION REQUIRED
            Your previous result was rejected for the following reason:
            {failure_context}

            Return a different result that avoids this problem.
        """
```

`run()` and `get_prompt()` are not declared on the base. Each subclass defines them with its own
typed arguments. There is no meaningful shared signature.

---

### 8.3 CourseAgent

- **Location**: `agents/course_agent.py`
- **Competency**: Proposals with no timeslot set.
- **Responsibility**: Propose the best timeslot for the course.
- **Output type**: `Proposal` (with `timeslot` populated)
- **Conflict reasoning**: reads all in-flight proposals and confirmed assignments from the board
  directly; filters for timeslot conflicts itself.

```python
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
            (p for p in board.get_proposals() if p.timeslot is None),
            None
        )

    def get_instruction(self) -> str:
        return """
            You are the CourseAgent in a timetable scheduling system.
            Your sole job is to propose the best timeslot for a given course.

            - Timeslot must be on a valid school day, within school hours, not during lunch
            - Avoid timeslots already taken by confirmed assignments
            - Avoid timeslots already claimed by other in-progress proposals
            - Spread courses across the week rather than clustering them

            Call log_decision to explain your reasoning. Return the proposal with timeslot set.
        """

    def get_prompt(self, proposal: Proposal, deps: Deps) -> str:
        course = next(c for c in deps.courses if c.id == proposal.course_id)

        confirmed_slots = [
            a.timeslot.model_dump()
            for a in deps.board.get_assignments()
        ]

        # agent reads raw proposals and computes its own conflict view
        in_flight_slots = [
            p.timeslot.model_dump()
            for p in deps.board.get_proposals()
            if p.id != proposal.id and p.timeslot is not None
        ]

        prompt = f"""
            Course:
            {json.dumps(course.model_dump(), indent=2)}

            School policy:
            {json.dumps(deps.policy.model_dump(), indent=2)}

            Confirmed assignments (timeslots taken):
            {json.dumps(confirmed_slots, indent=2)}

            Timeslots already claimed by other in-progress proposals (avoid these too):
            {json.dumps(in_flight_slots, indent=2)}
        """
        if proposal.failure_context:
            prompt += f"\n\n{self.get_failure_prompt(proposal.failure_context)}"
        return prompt

    async def run(self, proposal: Proposal, deps: Deps) -> None:
        prompt = self.get_prompt(proposal, deps)
        result = await self._agent.run(prompt, deps=deps, instructions=self.get_instruction())
        deps.board.update_proposal(result.data)
```

---

### 8.4 RoomAgent

- **Location**: `agents/room_agent.py`
- **Competency**: Proposals with a timeslot but no room_id.
- **Responsibility**: Assign the most suitable room.
- **Output type**: `Proposal` (with `room_id` populated)
- **Conflict reasoning**: reads all in-flight proposals and confirmed assignments; filters for
  room conflicts at the proposal's timeslot itself.

```python
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
            None
        )

    def get_instruction(self) -> str:
        return """
            You are the RoomAgent in a timetable scheduling system.
            Your sole job is to assign the most suitable room for a course proposal.

            - Match room type to course requirement (lab → lab room, non-lab → classroom)
            - Never assign a room already confirmed at the same timeslot
            - Never assign a room already claimed by another in-progress proposal at the same timeslot
            - If multiple suitable rooms are free, prefer the best fit

            Call log_decision to explain your reasoning. Return the proposal with room_id set.
        """

    def get_prompt(self, proposal: Proposal, deps: Deps) -> str:
        course = next(c for c in deps.courses if c.id == proposal.course_id)

        # rooms confirmed at this timeslot — agent reads raw assignments and filters itself
        confirmed_room_ids = [
            a.room_id
            for a in deps.board.get_assignments()
            if a.timeslot == proposal.timeslot
        ]

        # rooms claimed by other in-flight proposals at the same timeslot
        in_flight_room_ids = [
            p.room_id
            for p in deps.board.get_proposals()
            if p.id != proposal.id
            and p.timeslot == proposal.timeslot
            and p.room_id is not None
        ]

        prompt = f"""
            Current proposal:
            {json.dumps(proposal.model_dump(), indent=2)}

            Course:
            {json.dumps(course.model_dump(), indent=2)}

            All rooms:
            {json.dumps([r.model_dump() for r in deps.rooms], indent=2)}

            Room ids confirmed at this timeslot (unavailable):
            {json.dumps(confirmed_room_ids, indent=2)}

            Room ids claimed by other in-progress proposals at this timeslot (unavailable):
            {json.dumps(in_flight_room_ids, indent=2)}
        """
        if proposal.failure_context:
            prompt += f"\n\n{self.get_failure_prompt(proposal.failure_context)}"
        return prompt

    async def run(self, proposal: Proposal, deps: Deps) -> None:
        prompt = self.get_prompt(proposal, deps)
        result = await self._agent.run(prompt, deps=deps, instructions=self.get_instruction())
        deps.board.update_proposal(result.data)
```

---

### 8.5 LecturerAgent

- **Location**: `agents/lecturer_agent.py`
- **Competency**: Proposals with a room_id but no lecturer_id.
- **Responsibility**: Assign the most suitable lecturer.
- **Output type**: `Proposal` (with `lecturer_id` populated)
- **Conflict reasoning**: reads all in-flight proposals and confirmed assignments; filters for
  lecturer conflicts at the proposal's timeslot itself.

```python
import json
from pydantic_ai import Agent

from agents.base_agent import BaseAgent
from blackboard.blackboard import BlackBoard
from core.deps import Deps
from schemas.timetable import Proposal


class LecturerAgent(BaseAgent):
    def __init__(self, name: str, agent: Agent):
        super().__init__(name, agent)

    def is_competent_for(self, board: BlackBoard) -> Proposal | None:
        return next(
            (p for p in board.get_proposals()
             if p.room_id is not None and p.lecturer_id is None),
            None
        )

    def get_instruction(self) -> str:
        return """
            You are the LecturerAgent in a timetable scheduling system.
            Your sole job is to assign the most suitable lecturer for a course proposal.

            - Lecturer must teach this course (check courses_taught)
            - Lecturer must not already be confirmed at this timeslot
            - Lecturer must not have this timeslot in their unavailable_slots
            - Lecturer must not be claimed by another in-progress proposal at this timeslot
            - If multiple qualify, prefer the one with the lighter confirmed workload

            Call log_decision to explain your reasoning. Return the proposal with lecturer_id set.
        """

    def get_prompt(self, proposal: Proposal, deps: Deps) -> str:
        course = next(c for c in deps.courses if c.id == proposal.course_id)

        # lecturer ids confirmed at this timeslot — agent reads raw and filters itself
        confirmed_lecturer_ids = [
            a.lecturer_id
            for a in deps.board.get_assignments()
            if a.timeslot == proposal.timeslot
        ]

        # lecturer ids claimed by other in-flight proposals at the same timeslot
        in_flight_lecturer_ids = [
            p.lecturer_id
            for p in deps.board.get_proposals()
            if p.id != proposal.id
            and p.timeslot == proposal.timeslot
            and p.lecturer_id is not None
        ]

        prompt = f"""
            Current proposal:
            {json.dumps(proposal.model_dump(), indent=2)}

            Course:
            {json.dumps(course.model_dump(), indent=2)}

            All lecturers (includes courses_taught and unavailable_slots):
            {json.dumps([l.model_dump() for l in deps.lecturers], indent=2)}

            Lecturer ids confirmed at this timeslot (unavailable):
            {json.dumps(confirmed_lecturer_ids, indent=2)}

            Lecturer ids claimed by other in-progress proposals at this timeslot (unavailable):
            {json.dumps(in_flight_lecturer_ids, indent=2)}
        """
        if proposal.failure_context:
            prompt += f"\n\n{self.get_failure_prompt(proposal.failure_context)}"
        return prompt

    async def run(self, proposal: Proposal, deps: Deps) -> None:
        prompt = self.get_prompt(proposal, deps)
        result = await self._agent.run(prompt, deps=deps, instructions=self.get_instruction())
        deps.board.update_proposal(result.data)
```

---

### 8.6 PolicyAgent

- **Location**: `agents/policy_agent.py`
- **Competency**: Proposals that are fully assembled but not yet policy-evaluated.
- **Responsibility**: Check all policy rules. Set `policy_approved`, `policy_reason`, and
  `failed_component`. Write back to the board.
- **Implementation**: Pure Python — no LLM call.
- **Conflict reasoning**: reads raw proposals from the board and computes room and lecturer
  conflicts itself. No board helper methods.

```python
from blackboard.blackboard import BlackBoard
from agents.base_agent import BaseAgent
from core.deps import Deps
from schemas.timetable import Proposal


class PolicyAgent(BaseAgent):
    def __init__(self, name: str):
        self._name = name
        self._agent = None

    def is_competent_for(self, board: BlackBoard) -> Proposal | None:
        return next(
            (p for p in board.get_proposals()
             if p.lecturer_id is not None and p.policy_approved is None),
            None
        )

    def get_instruction(self) -> str:
        return ""

    def check(self, proposal: Proposal, deps: Deps) -> tuple[bool, str | None, str | None]:
        """
        Returns (approved, reason, failed_component).
        failed_component is one of "timeslot", "room", "lecturer", or None if approved.
        """
        policy   = deps.policy
        course   = next(c for c in deps.courses   if c.id == proposal.course_id)
        room     = next(r for r in deps.rooms     if r.id == proposal.room_id)
        lecturer = next(l for l in deps.lecturers if l.id == proposal.lecturer_id)
        slot     = proposal.timeslot

        if slot.day not in policy.school_days:
            return False, f"{slot.day} is not a valid school day", "timeslot"

        if slot.start_hour < policy.school_start_hour or slot.end_hour > policy.school_end_hour:
            return False, "Timeslot falls outside school hours", "timeslot"

        if slot.start_hour < policy.lunch_end_hour and slot.end_hour > policy.lunch_start_hour:
            return False, "Timeslot overlaps the lunch break", "timeslot"

        if course.requires_lab and room.room_type != "lab":
            return False, f"Course requires a lab but {room.name} is a {room.room_type}", "room"

        if proposal.course_id not in lecturer.courses_taught:
            return False, f"{lecturer.name} is not qualified to teach this course", "lecturer"

        if any(
            s.day == slot.day and s.start_hour == slot.start_hour
            for s in lecturer.unavailable_slots
        ):
            return False, f"{lecturer.name} is unavailable at this timeslot", "lecturer"

        # inter-proposal conflict checks — agent reads raw proposals itself
        room_conflicts = [
            p for p in deps.board.get_proposals()
            if p.id != proposal.id
            and p.room_id == proposal.room_id
            and p.timeslot == proposal.timeslot
        ]
        if room_conflicts:
            other = room_conflicts[0].course_id
            return False, f"{room.name} already claimed by in-flight proposal for {other}", "room"

        lecturer_conflicts = [
            p for p in deps.board.get_proposals()
            if p.id != proposal.id
            and p.lecturer_id == proposal.lecturer_id
            and p.timeslot == proposal.timeslot
        ]
        if lecturer_conflicts:
            other = lecturer_conflicts[0].course_id
            return False, f"{lecturer.name} already claimed by in-flight proposal for {other}", "lecturer"

        return True, None, None

    async def run(self, proposal: Proposal, deps: Deps) -> None:
        approved, reason, failed_component = self.check(proposal, deps)
        proposal.policy_approved  = approved
        proposal.policy_reason    = reason
        proposal.failed_component = failed_component
        deps.board.update_proposal(proposal)
```

---

## 9. Control Flow

**Location**: `control/scheduler.py`

### 9.1 Startup sequence

```
1. Load all JSON data from data/
2. Instantiate BlackBoard
3. Call board.seed([c.id for c in courses])
4. Instantiate Deps(board, courses, rooms, lecturers, policy, total_tokens=0)
5. Instantiate all four agents
6. Begin scheduler loop
```

### 9.2 Scheduler loop

The Scheduler is a thin activation loop with lifecycle responsibility. It has no domain knowledge.
It polls agents for competency, activates the first match, and handles proposal lifecycle
transitions — confirmation, abandonment, and rejection reset. It owns the reset logic that was
incorrectly placed in `Proposal.apply_rejection()` in v2.0.

```python
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

            for proposal in list(board.get_proposals()):
                if proposal.retry_count >= MAX_RETRIES:
                    logger.warning(f"[cycle {cycle}] abandoning {proposal.course_id}")
                    board.abandon_proposal(proposal, "Exceeded MAX_RETRIES", cycle)

                elif proposal.policy_approved is True:
                    logger.info(f"[cycle {cycle}] confirming {proposal.course_id}")
                    board.confirm_proposal(proposal, cycle)

                elif proposal.policy_approved is False:
                    logger.info(f"[cycle {cycle}] resetting {proposal.course_id} "
                                f"— failed: {proposal.failed_component}")
                    self._reset_proposal(proposal, board)

            board.ensure_proposals_exist(cycle)

            for agent in self._agents:
                if proposal := agent.is_competent_for(board):
                    logger.info(f"[cycle {cycle}] {agent.name} self-activated "
                                f"for {proposal.course_id}")
                    await agent.run(proposal, deps)
                    break

        return self._produce_output(deps, cycle)

    def _reset_proposal(self, proposal: Proposal, board: BlackBoard) -> None:
        """Reset proposal fields to the stage that failed so the correct agent re-activates."""
        if proposal.failed_component == "timeslot":
            proposal.timeslot    = None
            proposal.room_id     = None
            proposal.lecturer_id = None
        elif proposal.failed_component == "room":
            proposal.room_id     = None
            proposal.lecturer_id = None
        elif proposal.failed_component == "lecturer":
            proposal.lecturer_id = None

        proposal.failure_context  = proposal.policy_reason
        proposal.retry_count     += 1
        proposal.policy_approved  = None
        proposal.policy_reason    = None
        proposal.failed_component = None

        board.update_proposal(proposal)

    def _produce_output(self, deps: Deps, cycle: int) -> dict:
        # joins assignments with course, room, lecturer names from reference data
        # joins remaining unscheduled courses with rejection log for reason reporting
        ...
```

**What the Scheduler does:**
- Checks termination
- Abandons proposals that exceed `MAX_RETRIES`
- Confirms proposals the `PolicyAgent` has approved
- Resets rejected proposals via `_reset_proposal()` so the correct agent re-activates
- Creates new proposals for unscheduled courses
- Polls agents for competency and activates the first match

**What the Scheduler does not do:**
- It does not choose which agent handles domain work
- It does not interpret proposal content to make routing decisions
- It does not know what a timeslot, room, or lecturer is
- It does not pass context between agents

### 9.3 Termination

| Condition | Outcome |
|---|---|
| No unscheduled courses and no in-flight proposals | Success |
| Proposal `retry_count >= MAX_RETRIES` | That course abandoned |
| `cycle > MAX_CYCLES` | Partial result |

---

## 10. Core Infrastructure

### 10.1 Logger

**Location**: `core/logger.py`

```python
import logging
logger = logging.getLogger("timetable")
```

One named logger. All modules import this directly. The `log_decision` tool uses its own
`logging.getLogger(__name__)`.

### 10.2 LLM Factory

**Location**: `core/llm_factory.py`

```python
def make_model(provider: str) -> KnownModelName | Model:
    ...
```

Accepts a string like `"openai:gpt-4o-mini"` and returns a configured pydantic-ai model object.
Unchanged from v2.0.

### 10.3 Data Loader

**Location**: `core/data_loader.py`

```python
def load_data() -> tuple[list[Course], list[Room], list[Lecturer], Policy]:
    courses   = [Course.model_validate(c)   for c in json.loads(Path("data/courses.json").read_text())]
    rooms     = [Room.model_validate(r)     for r in json.loads(Path("data/rooms.json").read_text())]
    lecturers = [Lecturer.model_validate(l) for l in json.loads(Path("data/lecturers.json").read_text())]
    policy    = Policy.model_validate(json.loads(Path("data/policy.json").read_text()))
    return courses, rooms, lecturers, policy
```

Unchanged from v2.0.

---

## 11. Data Files

Identical to v2.0. No changes.

### courses.json
```json
[
  { "id": "CS201", "name": "Data Structures",             "requires_lab": true  },
  { "id": "MA101", "name": "Calculus",                    "requires_lab": false },
  { "id": "CS101", "name": "Introduction to Programming", "requires_lab": true  },
  { "id": "PH201", "name": "Physics I",                   "requires_lab": false }
]
```

### rooms.json
```json
[
  { "id": "R001", "name": "Lecture Theatre 1", "room_type": "classroom" },
  { "id": "R002", "name": "Lecture Theatre 2", "room_type": "classroom" },
  { "id": "R003", "name": "Computer Lab A",    "room_type": "lab"       },
  { "id": "R004", "name": "Computer Lab B",    "room_type": "lab"       }
]
```

### lecturers.json
```json
[
  {
    "id": "L001",
    "name": "Dr. Okafor",
    "courses_taught": ["CS201", "CS101"],
    "unavailable_slots": [
      { "day": "Monday", "start_hour": 8, "end_hour": 10 }
    ]
  },
  {
    "id": "L002",
    "name": "Prof. Singh",
    "courses_taught": ["MA101"],
    "unavailable_slots": []
  },
  {
    "id": "L003",
    "name": "Dr. Reyes",
    "courses_taught": ["PH201"],
    "unavailable_slots": [
      { "day": "Thursday", "start_hour": 14, "end_hour": 16 }
    ]
  }
]
```

### policy.json
```json
{
  "school_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
  "school_start_hour": 8,
  "school_end_hour": 17,
  "lunch_start_hour": 12,
  "lunch_end_hour": 13
}
```

---

## 12. Output Format

### Success
```json
{
  "generated": true,
  "total_cycles": 12,
  "total_tokens": 9840,
  "assignments": [
    {
      "course_id": "CS201",
      "course_name": "Data Structures",
      "room_id": "R003",
      "room_name": "Computer Lab A",
      "lecturer_id": "L001",
      "lecturer_name": "Dr. Okafor",
      "day": "Wednesday",
      "start_hour": 10,
      "end_hour": 11
    }
  ],
  "unresolved": []
}
```

### Partial or failure
```json
{
  "generated": false,
  "total_cycles": 50,
  "total_tokens": 31200,
  "assignments": [],
  "unresolved": [
    {
      "course_id": "CS201",
      "reason": "Abandoned after 5 retries. No lab available without lecturer conflict."
    }
  ]
}
```

---

## 13. main.py

```python
import asyncio
import json
from pydantic_ai import Agent
from blackboard.blackboard import BlackBoard
from control.scheduler import Scheduler
from core.data_loader import load_data
from core.deps import Deps
from core.logger import logger
from core.llm_factory import make_model
from agents.course_agent import CourseAgent
from agents.room_agent import RoomAgent
from agents.lecturer_agent import LecturerAgent
from agents.policy_agent import PolicyAgent
from tools.agent_logger import log_decision

PROVIDER = "openai:gpt-4o-mini"

async def main():
    courses, rooms, lecturers, policy = load_data()

    board = BlackBoard()
    board.seed([c.id for c in courses])

    deps = Deps(
        board=board,
        courses=courses,
        rooms=rooms,
        lecturers=lecturers,
        policy=policy,
        total_tokens=0,
    )

    agents = [
        CourseAgent(
            name="course",
            agent=Agent(model=make_model(PROVIDER), deps_type=Deps, output_type=Proposal, tools=[log_decision]),
        ),
        RoomAgent(
            name="room",
            agent=Agent(model=make_model(PROVIDER), deps_type=Deps, output_type=Proposal, tools=[log_decision]),
        ),
        LecturerAgent(
            name="lecturer",
            agent=Agent(model=make_model(PROVIDER), deps_type=Deps, output_type=Proposal, tools=[log_decision]),
        ),
        PolicyAgent(name="policy"),
    ]

    scheduler = Scheduler(agents=agents)
    result = await scheduler.run(deps)
    print(json.dumps(result, indent=2))

asyncio.run(main())
```

Unchanged from v2.0.

---

## 14. MAS Patterns Applied

### Coordination Patterns
- **Blackboard** — `BlackBoard` is the shared knowledge base; all agents read and write directly;
  the board holds raw state only, no domain logic
- **Competency-Based Role** — each agent defines its own activation condition (`is_competent_for`);
  role = domain competency + guard condition + reaction, not just a label
- **Sequential Pipeline (emergent)** — timeslot → room → lecturer → policy order is not declared;
  it emerges from each agent's guard conditions on proposal field readiness
- **Planner-Generator-Evaluator** — scheduler resets failed fields (plan), agent regenerates with
  `failure_context` (generate), `PolicyAgent` re-evaluates (evaluate); loop until approved or abandoned

### Communication Mechanism
- **Blackboard (read/write)** — agents communicate exclusively by reading from and writing to the
  shared `BlackBoard`; no direct agent-to-agent calls; no message passing

### Responsibility boundaries (v3.0 clarification)

| Component | Owns |
|---|---|
| `BlackBoard` | Raw state storage and retrieval only |
| `Proposal` | Data shape only — no methods, no mutation |
| Each Agent | Its own conflict reasoning, context assembly, domain decision |
| `Scheduler` | Proposal lifecycle: confirm, abandon, reset on rejection |

---

## 15. Implementation Sequence

Build and verify in this order.

**Phase 1 — Schemas** (`schemas/`)
All Pydantic models. Verify by instantiating each with dummy data and round-tripping through
`model_dump()` / `model_validate()`. Confirm `Proposal` has no methods beyond Pydantic defaults.

**Phase 2 — BlackBoard** (`blackboard/blackboard.py`)
Seed, add_proposal, update_proposal, confirm_proposal, abandon_proposal, all four read methods.
Verify with a standalone script: seed three courses, add two proposals, update one, confirm one,
abandon one. Confirm no conflict methods exist on the board.

**Phase 3 — Core infrastructure** (`core/`)
`deps.py`, `llm_factory.py`, `data_loader.py`, `logger.py`. Verify by loading data files and
printing parsed models.

**Phase 4 — Data files** (`data/`)
Copy the four JSON files from section 11. Verify by running `load_data()`.

**Phase 5 — Tools** (`tools/agent_logger.py`)
The single `log_decision` tool. Verify by instantiating a minimal pydantic-ai agent with this tool
and confirming a log line appears with a non-zero token count.

**Phase 6 — PolicyAgent** (`agents/policy_agent.py`)
Build first — no LLM dependency. Verify `check()` directly: test each failure branch (invalid day,
outside hours, lunch overlap, wrong room type, unqualified lecturer, unavailable lecturer,
inter-proposal room conflict, inter-proposal lecturer conflict) and the success path. Confirm
`failed_component` is set correctly for each branch. Confirm all conflict detection reads raw
proposals from the board, not from board helper methods.

**Phase 7 — Worker agents** (`agents/`)
Build in order: `RoomAgent` → `LecturerAgent` → `CourseAgent`. For each: verify
`is_competent_for()` returns the correct proposal given a board with proposals in various states.
Verify `get_prompt()` reads raw board state and assembles its own conflict view. Call `run()` with
a minimal board and deps and confirm the agent writes an updated proposal back to the board.

**Phase 8 — Scheduler** (`control/scheduler.py`)
Build after all agents are verified. Test `_reset_proposal()` in isolation for each
`failed_component` value. Test the full activation loop: construct a board with proposals at each
stage and confirm the correct agent self-activates each cycle. Test happy path end-to-end. Test
failure recovery by making a constraint impossible and confirming abandonment after MAX_RETRIES.

**Phase 9 — Integration** (`main.py`)
Run end-to-end. Confirm: no orchestrator called, agents self-activate in correct order,
`log_decision` lines appear with non-zero tokens for LLM agents, `PolicyAgent` produces no token
lines, final output includes `total_tokens` from the three LLM agents only.