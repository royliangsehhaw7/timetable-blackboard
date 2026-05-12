# Multi-Agent Timetable Generation System
## Specification v2.0 — Blackboard / Competency-Based Role Pattern

---

## 1. Purpose

A learning-oriented multi-agent system that generates a valid weekly school timetable. The primary
goal is to understand and implement the **Blackboard Pattern with Competency-Based Roles** in a
grounded, practical context using pydantic-ai's deps, agents, and tools features properly.

The timetable domain is intentionally simple. It exists to make emergent self-activation visible and
traceable, not to solve a hard scheduling problem.

**What is different from v1.0 (Orchestrator-Worker):**

In v1.0, a central orchestrator read full state every cycle and told each worker when to act.
Workers were passive — they waited to be dispatched.

In v2.0, there is no orchestrator. Each agent owns its activation condition. The blackboard holds
all shared state. A thin scheduler loop asks every agent every cycle whether its preconditions are
met. If they are, the agent activates itself, does its domain work, and writes back to the
blackboard. The sequential order of scheduling — timeslot before room, room before lecturer,
lecturer before policy — is preserved, but it emerges from data readiness on the blackboard, not
from an external dispatcher.

---

## 2. Architecture

### 2.1 Pattern

This system implements the **Blackboard Pattern with Competency-Based Roles**.

A shared `BlackBoard` object holds all system state — proposals in progress, confirmed assignments,
rejection history, and unscheduled course ids. Every agent reads directly from the blackboard and
writes directly back to it. No coordinator mediates these reads or writes.

Each agent defines its own **competency guard** — a method that inspects the blackboard and returns
the proposal it should work on, or `None` if no work is currently relevant. The scheduler loop
calls every agent's guard every cycle. Whichever agent finds relevant work activates itself. The
rest stay idle.

The sequential constraint of the scheduling pipeline — timeslot → room → lecturer → policy — is
enforced by each agent's guard condition, not by external dispatch:

- `CourseAgent` activates when a proposal exists with no timeslot
- `RoomAgent` activates when a proposal has a timeslot but no room
- `LecturerAgent` activates when a proposal has a room but no lecturer
- `PolicyAgent` activates when a proposal is fully assembled but unvalidated

No agent is told to go. Each agent decides for itself whether it is relevant to the current state.

### 2.2 The Blackboard

The `BlackBoard` is the single source of truth. It replaces the `Store` from v1.0 and expands it
to include in-progress proposals. All agents read from it. All agents write to it. The scheduler
loop also writes to it (confirm, abandon).

The blackboard is not a message bus. Agents do not send messages to each other. They leave state on
the board and trust that the relevant agent will notice and act.

### 2.3 Agents

There are four pydantic-ai agents. The orchestrator from v1.0 is removed entirely.

| Agent | Competency — activates when |
|---|---|
| `CourseAgent` | A proposal exists on the board with no timeslot set |
| `RoomAgent` | A proposal exists with a timeslot but no room_id |
| `LecturerAgent` | A proposal exists with a room_id but no lecturer_id |
| `PolicyAgent` | A proposal exists that is fully assembled but policy not yet evaluated |

All four share a common `BaseAgent`. The `PolicyAgent` is a Python function — not an LLM call —
because policy checking is fully deterministic (see section 8.5).

### 2.4 Flow

Every cycle follows the same structure: the scheduler reads the board, asks each agent if it has
work, activates the first one that does, and repeats.

```
LOOP:
    board state snapshot logged

    if no unscheduled courses and no active proposals → done

    if active proposal retry_count >= MAX_RETRIES → abandon proposal, write to board

    for each agent in [CourseAgent, RoomAgent, LecturerAgent, PolicyAgent]:
        proposal = agent.is_competent_for(board)
        if proposal:
            agent.run(proposal, board, deps)   # agent reads board, writes back to board
            break                              # one agent activates per cycle

    if no agent activated and unscheduled courses remain:
        create new Proposal for next unscheduled course, write to board
```

**What each agent's guard checks:**

- `CourseAgent.is_competent_for(board)` — finds any proposal where `timeslot is None`
- `RoomAgent.is_competent_for(board)` — finds any proposal where `timeslot is set` and `room_id is None`
- `LecturerAgent.is_competent_for(board)` — finds any proposal where `room_id is set` and `lecturer_id is None`
- `PolicyAgent.is_competent_for(board)` — finds any proposal where `lecturer_id is set` and `policy_approved is None`

**What happens after each agent runs:**

- `CourseAgent` — writes proposal with `timeslot` set back to board
- `RoomAgent` — writes proposal with `room_id` set back to board
- `LecturerAgent` — writes proposal with `lecturer_id` set back to board
- `PolicyAgent` — writes proposal with `policy_approved` and `policy_reason` set back to board; if approved, scheduler confirms it; if rejected, scheduler increments `retry_count` and resets the responsible field so the right agent re-activates

### 2.5 Failure Recovery

When `PolicyAgent` rejects a proposal it sets `policy_approved=False`, `policy_reason` with a
specific failure description, and `failed_component` with one of `"timeslot"`, `"room"`,
`"lecturer"`. The scheduler reads `failed_component` and resets the corresponding field on the
proposal to `None`, then increments `retry_count`. On the next cycle, the agent responsible for
that field will find the proposal via its guard and re-activate with the `failure_context` from
`policy_reason`.

This is not dispatch. The scheduler does not choose which agent to call. It resets a field. The
agent whose guard matches that field self-activates on the next cycle.

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

**Differences from v1.0:**

- `store/store.py` is removed — replaced by `blackboard/blackboard.py`
- `control/coordinator.py` is removed — replaced by `control/scheduler.py`
- `agents/orchestrator_agent.py` is removed entirely
- All other files retain the same location

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

    # ── state predicates — the scheduler reads only these ─────────────────────

    @property
    def is_confirmed(self) -> bool:
        return self.policy_approved is True

    @property
    def is_rejected(self) -> bool:
        return self.policy_approved is False

    @property
    def is_exhausted(self) -> bool:
        return self.retry_count >= MAX_RETRIES

    # ── self-repair — proposal owns its own field topology ────────────────────

    def apply_rejection(self) -> None:
        """
        Reset this proposal back to the stage that failed.
        The scheduler calls this without knowing what it does internally.
        The correct agent self-activates on the next cycle because its
        guard condition will match the reset state.
        """
        component = self.failed_component

        self.failure_context  = self.policy_reason
        self.retry_count     += 1
        self.policy_approved  = None
        self.policy_reason    = None
        self.failed_component = None

        _RESET_STRATEGY[component](self)

    def _reset_timeslot(p: Proposal) -> None:
        p.timeslot    = None
        p.room_id     = None
        p.lecturer_id = None

    def _reset_room(p: Proposal) -> None:
        p.room_id     = None
        p.lecturer_id = None

    def _reset_lecturer(p: Proposal) -> None:
        p.lecturer_id = None

    _RESET_STRATEGY: dict[str | None, callable] = {
        "timeslot": _reset_timeslot,
        "room":     _reset_room,
        "lecturer": _reset_lecturer,
}
```

**Changes from v1.0:**

- `OrchestratorDecision` is removed — no orchestrator exists
- `Proposal` gains two fields:
  - `failed_component` — the structured rejection target set by `PolicyAgent` (`"timeslot"`, `"room"`, or `"lecturer"`). Enables the scheduler to reset the correct field without LLM reasoning.
  - `failure_context` — free-text context copied from `policy_reason` by the scheduler and passed to the re-activating agent as a correction hint.

**Verification**: import each schema in a scratch script and instantiate one instance per model with
dummy data. All fields should round-trip through `model_dump()` and `model_validate()` without
error. Confirm `failed_component` and `failure_context` default to `None`.

**Invariant**: the scheduler must never read policy_approved, retry_count, failed_component,
timeslot, room_id, or lecturer_id directly. It reads only is_confirmed, is_rejected, and
is_exhausted. All field knowledge lives inside Proposal.

---

## 5. The BlackBoard

**Location**: `blackboard/blackboard.py`

The blackboard is the shared knowledge base. It replaces the `Store` from v1.0. All agents read
from it directly and write to it directly. The scheduler also reads and writes to it for lifecycle
operations (confirm, abandon, new proposal creation).

Unlike the v1.0 Store — which was a write-only surface for the coordinator — the blackboard is a
read/write surface for every participant in the system. This is the defining characteristic of the
Blackboard pattern.

### 5.1 What it holds

```python
class BlackBoard:
    def __init__(self):
        self._proposals: dict[str, Proposal] = {}         # keyed by proposal.id
        self._assignments: list[Assignment] = []
        self._rejection_log: list[RejectionRecord] = []
        self._unscheduled_courses: list[str] = []
```

`_proposals` holds all in-progress proposals. In v1.0, in-flight proposals lived as a local
variable in the coordinator loop. In v2.0, they live on the blackboard, visible to every agent.
Multiple proposals can coexist on the board (one per unscheduled course), though in this system
only one is in progress at any time.

`_unscheduled_courses` is seeded at startup. A course id is removed when confirmed or abandoned.

### 5.2 Methods

#### Seeding

| Method | Signature | What it does |
|---|---|---|
| `seed` | `(course_ids: list[str]) -> None` | Populates `_unscheduled_courses` at startup. Called once before the scheduler loop begins. |

#### Proposal lifecycle

| Method | Signature | What it does |
|---|---|---|
| `add_proposal` | `(proposal: Proposal) -> None` | Adds a new in-progress proposal to the board. Called by the scheduler when starting a new course. |
| `update_proposal` | `(proposal: Proposal) -> None` | Replaces the existing proposal with the same id. Called by agents after they complete their domain work. |
| `confirm_proposal` | `(proposal: Proposal, cycle: int) -> None` | Constructs an `Assignment`, appends to `_assignments`, removes `course_id` from `_unscheduled_courses`, removes proposal from `_proposals`. |
| `abandon_proposal` | `(proposal: Proposal, reason: str, cycle: int) -> None` | Appends a `RejectionRecord`, removes `course_id` from `_unscheduled_courses`, removes proposal from `_proposals`. |

#### Read methods

| Method | Signature | What it returns |
|---|---|---|
| `get_proposals` | `() -> list[Proposal]` | All in-progress proposals. |
| `get_assignments` | `() -> list[Assignment]` | All confirmed assignments. |
| `get_rejection_log` | `() -> list[RejectionRecord]` | Full rejection history. |
| `get_unscheduled_courses` | `() -> list[str]` | Course ids not yet confirmed or abandoned. |

No domain logic lives in the blackboard. No routing. No filtering by agent. Just reads and writes.

**Verification**: seed with three course ids. Add two proposals. Update one. Confirm one proposal
and verify it appears in assignments and is removed from proposals and unscheduled. Abandon one and
verify rejection log is updated. Print all collections after each operation.

---

## 6. Dependencies — Deps and RunContext

### 6.1 What Deps holds

`Deps` is a single container passed into every agent activation. It holds the blackboard (shared
mutable state) and the reference data (read-only).

```python
@dataclass
class Deps:
    board: BlackBoard            # mutable — agents read and write directly
    courses: list[Course]        # read-only reference data
    rooms: list[Room]            # read-only reference data
    lecturers: list[Lecturer]    # read-only reference data
    policy: Policy               # read-only reference data
    total_tokens: int = 0        # accumulated token usage across all agents
```

**Change from v1.0:** `store: Store` is replaced by `board: BlackBoard`. Agents now have direct
access to the blackboard through `deps.board`. This is intentional — agents in a blackboard system
read and write shared state themselves, they do not receive pre-filtered context from a coordinator.

### 6.2 How Deps flows at runtime

One `Deps` instance is created before the scheduler loop and passed into every agent's `run()`
call. pydantic-ai injects it into `RunContext` so the `log_decision` tool can accumulate token
usage into `deps.total_tokens`.

---

## 7. Tools

### 7.1 Design decision

There is one tool in this system: `log_decision`. It is registered on all LLM-powered agents.

A tool is something the LLM calls mid-reasoning to perform an action with a side effect.
`log_decision` qualifies: the LLM triggers it deliberately to narrate its reasoning, and it
produces structured log output and accumulates token counts as side effects.

Blackboard writes (update_proposal, confirm_proposal, abandon_proposal) happen inside each agent's
`run()` method or inside the scheduler after reading the returned proposal. They are not tools
because they are deterministic Python operations, not LLM-triggered mid-reasoning actions.

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

`log_decision` is registered on the three LLM-powered agents. `PolicyAgent` is a Python function
and does not use it.

| Tool | CourseAgent | RoomAgent | LecturerAgent | PolicyAgent |
|---|:---:|:---:|:---:|:---:|
| `log_decision` | ✓ | ✓ | ✓ | — |

---

## 8. Agents

### 8.1 Design principles

Each agent has one domain responsibility and one competency guard. The guard is a pure Python
method — no LLM involved. The domain work is LLM-powered (except PolicyAgent which is deterministic
Python).

Each agent has three methods:

- `is_competent_for(board)` — inspects the blackboard, returns the first proposal this agent
  should work on, or `None`. This is the agent's self-activation condition. It is called by the
  scheduler every cycle.
- `get_instruction()` — returns the agent's stable identity and rules. Passed to `agent.run()` as
  `instructions=`, which pydantic-ai sends as the system turn.
- `get_prompt(proposal, deps)` — returns the live situational data for this specific activation.
  Reads from the blackboard via `deps.board`. Passed as the user turn.

Each agent's `run()` method calls the LLM, receives a structured `Proposal` back, and writes it to
the blackboard via `deps.board.update_proposal(proposal)`. The agent writes to the board itself —
the scheduler does not do this on its behalf.

The sequential order of the pipeline is not declared anywhere explicitly. It emerges from the guard
conditions: `CourseAgent` only finds proposals with no timeslot, so it always activates first.
`RoomAgent` only finds proposals with a timeslot, so it always activates second. And so on.

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

    @abstractmethod
    def is_competent_for(self, board: BlackBoard) -> Proposal | None:
        """Inspect the board and return the proposal this agent should work on,
        or None if no current work is relevant to this agent's competency."""
        ...

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

**Key points:**

- `is_competent_for()` is abstract — every subclass must implement it. This is the defining method
  of the Blackboard / Competency-Based Role pattern. It replaces external dispatch entirely.
- `get_instruction()` is abstract — every subclass must implement it.
- `get_prompt()` is **not** declared on the base. Each subclass defines it with its own typed
  arguments. There is no meaningful shared signature.
- `run()` is **not** declared on the base. Each agent's `run()` takes different arguments and has
  different write-back behaviour.
- `get_failure_prompt()` is concrete and shared. It formats the `failure_context` passed from the
  scheduler when an agent re-activates after a policy rejection.

---

### 8.3 CourseAgent

- **Location**: `agents/course_agent.py`
- **Competency**: Proposals with no timeslot set.
- **Responsibility**: Propose the best timeslot for the course.
- **Output type**: `Proposal` (with `timeslot` populated)

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
            - Spread courses across the week rather than clustering them

            Call log_decision to explain your reasoning. Return the proposal with timeslot set.
        """

    def get_prompt(self, proposal: Proposal, deps: Deps) -> str:
        course = next(c for c in deps.courses if c.id == proposal.course_id)
        assignments = deps.board.get_assignments()
        prompt = f"""
            Course:
            {json.dumps(course.model_dump(), indent=2)}

            School policy:
            {json.dumps(deps.policy.model_dump(), indent=2)}

            Already confirmed assignments (timeslots already taken):
            {json.dumps([a.model_dump() for a in assignments], indent=2)}
        """
        if proposal.failure_context:
            prompt += f"\n\n{self.get_failure_prompt(proposal.failure_context)}"
        return prompt

    async def run(self, proposal: Proposal, deps: Deps) -> None:
        prompt = self.get_prompt(proposal, deps)
        result = await self._agent.run(prompt, deps=deps, instructions=self.get_instruction())
        updated = result.data
        deps.board.update_proposal(updated)
```

**Key difference from v1.0:** `run()` returns `None`. The agent writes the updated proposal
directly to the blackboard via `deps.board.update_proposal()`. The scheduler does not handle
write-back. The agent owns its own output.

`failure_context` is read directly from the proposal on the board — the scheduler wrote it there
after the policy rejection. The agent does not receive it as a parameter; it finds it on its
proposal.

---

### 8.4 RoomAgent

- **Location**: `agents/room_agent.py`
- **Competency**: Proposals with a timeslot but no room_id.
- **Responsibility**: Assign the most suitable room.
- **Output type**: `Proposal` (with `room_id` populated)

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
            - If multiple suitable rooms are free, prefer the best fit

            Call log_decision to explain your reasoning. Return the proposal with room_id set.
        """

    def get_prompt(self, proposal: Proposal, deps: Deps) -> str:
        course = next(c for c in deps.courses if c.id == proposal.course_id)
        booked_at_slot = [
            a for a in deps.board.get_assignments()
            if a.timeslot.day == proposal.timeslot.day
            and a.timeslot.start_hour == proposal.timeslot.start_hour
        ]
        prompt = f"""
            Current proposal:
            {json.dumps(proposal.model_dump(), indent=2)}

            Course:
            {json.dumps(course.model_dump(), indent=2)}

            All rooms:
            {json.dumps([r.model_dump() for r in deps.rooms], indent=2)}

            Rooms already confirmed at this timeslot:
            {json.dumps([a.model_dump() for a in booked_at_slot], indent=2)}
        """
        if proposal.failure_context:
            prompt += f"\n\n{self.get_failure_prompt(proposal.failure_context)}"
        return prompt

    async def run(self, proposal: Proposal, deps: Deps) -> None:
        prompt = self.get_prompt(proposal, deps)
        result = await self._agent.run(prompt, deps=deps, instructions=self.get_instruction())
        updated = result.data
        deps.board.update_proposal(updated)
```

---

### 8.5 LecturerAgent

- **Location**: `agents/lecturer_agent.py`
- **Competency**: Proposals with a room_id but no lecturer_id.
- **Responsibility**: Assign the most suitable lecturer.
- **Output type**: `Proposal` (with `lecturer_id` populated)

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
            - If multiple qualify, prefer the one with the lighter confirmed workload

            Call log_decision to explain your reasoning. Return the proposal with lecturer_id set.
        """

    def get_prompt(self, proposal: Proposal, deps: Deps) -> str:
        course = next(c for c in deps.courses if c.id == proposal.course_id)
        booked_at_slot = [
            a for a in deps.board.get_assignments()
            if a.timeslot.day == proposal.timeslot.day
            and a.timeslot.start_hour == proposal.timeslot.start_hour
        ]
        prompt = f"""
            Current proposal:
            {json.dumps(proposal.model_dump(), indent=2)}

            Course:
            {json.dumps(course.model_dump(), indent=2)}

            All lecturers (includes courses_taught and unavailable_slots):
            {json.dumps([l.model_dump() for l in deps.lecturers], indent=2)}

            Lecturers already confirmed at this timeslot:
            {json.dumps([a.model_dump() for a in booked_at_slot], indent=2)}
        """
        if proposal.failure_context:
            prompt += f"\n\n{self.get_failure_prompt(proposal.failure_context)}"
        return prompt

    async def run(self, proposal: Proposal, deps: Deps) -> None:
        prompt = self.get_prompt(proposal, deps)
        result = await self._agent.run(prompt, deps=deps, instructions=self.get_instruction())
        updated = result.data
        deps.board.update_proposal(updated)
```

---

### 8.6 PolicyAgent

- **Location**: `agents/policy_agent.py`
- **Competency**: Proposals that are fully assembled but not yet policy-evaluated.
- **Responsibility**: Check all policy rules. Set `policy_approved`, `policy_reason`, and
  `failed_component`. Write back to the board.
- **Implementation**: Pure Python function — no LLM call.

Policy checking is entirely deterministic. Every rule is a boolean condition on structured data.
Using an LLM for this adds latency, cost, and the risk of arithmetic errors on hour comparisons.
A Python function is faster, cheaper, and strictly more reliable.

```python
from blackboard.blackboard import BlackBoard
from agents.base_agent import BaseAgent
from core.deps import Deps
from schemas.timetable import Proposal

class PolicyAgent(BaseAgent):
    def __init__(self, name: str):
        self._name = name
        self._agent = None      # no LLM — policy checking is deterministic Python

    def is_competent_for(self, board: BlackBoard) -> Proposal | None:
        return next(
            (p for p in board.get_proposals()
             if p.lecturer_id is not None and p.policy_approved is None),
            None
        )

    def get_instruction(self) -> str:
        return ""               # not used — no LLM

    def check(self, proposal: Proposal, deps: Deps) -> tuple[bool, str | None, str | None]:
        """
        Returns (approved, reason, failed_component).
        failed_component is one of "timeslot", "room", "lecturer", or None if approved.
        """
        policy  = deps.policy
        course  = next(c for c in deps.courses  if c.id == proposal.course_id)
        room    = next(r for r in deps.rooms    if r.id == proposal.room_id)
        lecturer = next(l for l in deps.lecturers if l.id == proposal.lecturer_id)
        slot    = proposal.timeslot

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

        return True, None, None

    async def run(self, proposal: Proposal, deps: Deps) -> None:
        approved, reason, failed_component = self.check(proposal, deps)
        proposal.policy_approved  = approved
        proposal.policy_reason    = reason
        proposal.failed_component = failed_component
        deps.board.update_proposal(proposal)
```

**Note on `BaseAgent` inheritance:** `PolicyAgent` inherits `BaseAgent` to satisfy the scheduler's
uniform interface — `is_competent_for()` is still defined and still called by the scheduler. The
`agent` field is `None` because no pydantic-ai `Agent` is needed. `get_instruction()` returns an
empty string and is never called.

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

The scheduler is a thin activation loop. It has no domain knowledge and makes no routing decisions.
It asks each agent whether it is competent, activates the first one that is, and handles lifecycle
operations (new proposal creation, confirmation, abandonment) that are not agent domain work.

```python
MAX_CYCLES = 50
MAX_RETRIES = 5

class Scheduler:
    def __init__(self, agents: list[BaseAgent]):
        self._agents = agents   # ordered: [CourseAgent, RoomAgent, LecturerAgent, PolicyAgent]

    async def run(self, deps: Deps) -> dict:
        cycle = 0

        while cycle < MAX_CYCLES:
            cycle += 1
            board = deps.board
            logger.info(f"[cycle {cycle}] unscheduled: {board.get_unscheduled_courses()}")
            logger.info(f"[cycle {cycle}] proposals in flight: {[p.id for p in board.get_proposals()]}")

            # termination — nothing left to schedule
            if not board.get_unscheduled_courses() and not board.get_proposals():
                logger.info(f"[cycle {cycle}] all courses scheduled — done")
                break

            # abandon any proposal that has exceeded MAX_RETRIES
            for proposal in board.get_proposals():
                if proposal.retry_count >= MAX_RETRIES:
                    logger.warning(f"[cycle {cycle}] abandoning {proposal.course_id} after {proposal.retry_count} retries")
                    board.abandon_proposal(proposal, "Exceeded MAX_RETRIES", cycle)

            # handle approved proposals — confirm them
            for proposal in board.get_proposals():
                if proposal.policy_approved is True:
                    logger.info(f"[cycle {cycle}] confirming {proposal.course_id}")
                    board.confirm_proposal(proposal, cycle)

            # handle rejected proposals — reset the failed field, increment retry_count
            for proposal in board.get_proposals():
                if proposal.policy_approved is False:
                    logger.info(
                        f"[cycle {cycle}] policy rejected {proposal.course_id}: "
                        f"{proposal.policy_reason} — resetting {proposal.failed_component}"
                    )
                    proposal.failure_context = proposal.policy_reason
                    proposal.retry_count    += 1
                    proposal.policy_approved = None
                    proposal.policy_reason   = None
                    # reset the failed field so the responsible agent re-activates
                    if proposal.failed_component == "timeslot":
                        proposal.timeslot    = None
                        proposal.room_id     = None
                        proposal.lecturer_id = None
                    elif proposal.failed_component == "room":
                        proposal.room_id     = None
                        proposal.lecturer_id = None
                    elif proposal.failed_component == "lecturer":
                        proposal.lecturer_id = None
                    proposal.failed_component = None
                    board.update_proposal(proposal)

            # start a new proposal if a course has no proposal in flight yet
            scheduled_ids = {p.course_id for p in board.get_proposals()}
            for course_id in board.get_unscheduled_courses():
                if course_id not in scheduled_ids:
                    import uuid
                    new_proposal = Proposal(id=str(uuid.uuid4()), course_id=course_id)
                    board.add_proposal(new_proposal)
                    logger.info(f"[cycle {cycle}] new proposal created for {course_id}")
                    break   # one new proposal per cycle

            # ask each agent if it is competent — activate the first match
            activated = False
            for agent in self._agents:
                proposal = agent.is_competent_for(board)
                if proposal:
                    logger.info(f"[cycle {cycle}] {agent._name} self-activated for {proposal.course_id}")
                    await agent.run(proposal, deps)
                    activated = True
                    break

            if not activated:
                logger.info(f"[cycle {cycle}] no agent activated this cycle")

        return self._produce_output(deps, cycle)
```

**What the scheduler does:**

- Checks termination
- Abandons proposals that exceed `MAX_RETRIES`
- Confirms proposals the `PolicyAgent` has approved
- Resets rejected proposals so the correct agent re-activates
- Creates new proposals for unscheduled courses
- Polls agents for competency and activates the first match

**What the scheduler does not do:**

- It does not choose which agent to call for domain work
- It does not interpret proposal state to make routing decisions
- It does not know what a timeslot, room, or lecturer is
- It does not pass context between agents

### 9.3 Termination

| Condition | Outcome |
|---|---|
| No unscheduled courses and no in-flight proposals | Success — full timetable produced |
| Proposal `retry_count >= MAX_RETRIES` | That course abandoned, removed from board |
| `cycle > MAX_CYCLES` | Partial result — report what was scheduled and what remains |

### 9.4 Output

```python
def _produce_output(self, deps: Deps, cycle: int) -> dict:
    # joins assignments with course, room, lecturer names from reference data
    # joins remaining unscheduled courses with rejection log for reason reporting
```

---

## 10. Core Infrastructure

### 10.1 Logger

**Location**: `core/logger.py`

```python
import logging

logger = logging.getLogger("timetable")
```

One named logger. All modules import this and call `logger.info(...)` or `logger.warning(...)`
directly at the call site. No wrapper functions.

The `log_decision` tool in `tools/agent_logger.py` uses its own `logging.getLogger(__name__)` as
before.

### 10.2 LLM Factory

**Location**: `core/llm_factory.py`

```python
def make_model(provider: str) -> KnownModelName | Model:
    ...
```

Accepts a string like `"openai:gpt-4o-mini"` or `"anthropic:claude-haiku-3-5"` and returns a
configured pydantic-ai model object. Unchanged from v1.0.

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

Unchanged from v1.0.

---

## 11. Data Files

Identical to v1.0. No changes.

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

Identical to v1.0.

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

**Differences from v1.0:**

- `OrchestratorAgent` import and instantiation removed entirely
- `Store` replaced by `BlackBoard`
- `Coordinator` replaced by `Scheduler`
- `agents` is now a `list` ordered by activation priority, not a `dict` keyed by name —
  order matters because the scheduler polls them in sequence
- `PolicyAgent` receives only `name` — no pydantic-ai `Agent` needed
- Provider defaults to `gpt-4o-mini` — smaller model appropriate for these tasks

---

## 14. MAS Patterns Applied

```markdown
## MAS Patterns Applied

### Coordination Patterns
- Blackboard — BlackBoard is the shared knowledge base; all agents read and write directly;
  no coordinator mediates access
- Competency-Based Role — each agent defines its own activation condition (is_competent_for);
  role = domain competency + guard condition + reaction, not just a label
- Sequential Pipeline (emergent) — timeslot → room → lecturer → policy order is not declared;
  it emerges from each agent's guard conditions on proposal field readiness
- Planner-Generator-Evaluator — scheduler resets failed fields (plan), agent regenerates with
  failure_context (generate), PolicyAgent re-evaluates (evaluate); loop until approved or abandoned

### Communication Mechanism
- Blackboard (read/write) — agents communicate exclusively by reading from and writing to the
  shared BlackBoard; no direct agent-to-agent calls; no message passing
```

---

## 15. Implementation Sequence

Build and verify in this order. Each phase has a verification step — do not proceed to the next
phase until verification passes.

**Phase 1 — Schemas** (`schemas/`)
All Pydantic models. Verify by instantiating each with dummy data and round-tripping through
`model_dump()` / `model_validate()`. Confirm `failed_component` and `failure_context` default to
`None` on `Proposal`.

**Phase 2 — BlackBoard** (`blackboard/blackboard.py`)
Seed, add_proposal, update_proposal, confirm_proposal, abandon_proposal, all four reads. Verify
with a standalone script: seed three courses, add two proposals, update one, confirm one, abandon
one. Print all collections after each operation and confirm state is correct.

**Phase 3 — Core infrastructure** (`core/`)
`deps.py`, `llm_factory.py`, `data_loader.py`, `logger.py`. Verify by loading data files and
printing parsed models. Confirm `Deps.board` accepts a `BlackBoard` instance.

**Phase 4 — Data files** (`data/`)
Copy the four JSON files from section 11. Verify by running `load_data()` and confirming all
models parse without error.

**Phase 5 — Tools** (`tools/agent_logger.py`)
The single `log_decision` tool. Verify by instantiating a minimal pydantic-ai agent with this tool
and confirming a log line appears with a non-zero token count on first call.

**Phase 6 — PolicyAgent** (`agents/policy_agent.py`)
Build first because it has no LLM dependency and its `check()` method is directly testable. Verify
by calling `check()` directly with a manually constructed proposal, course, room, lecturer, and
policy — test each failure branch (invalid day, outside hours, lunch overlap, wrong room type,
unqualified lecturer, unavailable lecturer) and the success path. Confirm `failed_component` is set
correctly for each branch.

**Phase 7 — Worker agents** (`agents/`)
Build in this order: `RoomAgent` → `LecturerAgent` → `CourseAgent`. For each: verify
`is_competent_for()` returns the correct proposal given a board with proposals in various states,
and returns `None` when no proposal matches its guard. Then call `run()` with a minimal board and
deps, confirm the agent writes an updated proposal back to the board with the expected field
populated.

**Phase 8 — Scheduler** (`control/scheduler.py`)
Build after all agents are verified. Test the activation loop in isolation: construct a board with
proposals at each stage and confirm the correct agent self-activates each cycle. Then test the full
happy path end-to-end — all four courses should confirm. Then test failure recovery by temporarily
making a constraint impossible (e.g. remove all lab rooms) and confirming proposals are abandoned
after MAX_RETRIES.

**Phase 9 — Integration** (`main.py`)
Run end-to-end. Observe log output to confirm: no orchestrator is called, agents self-activate
in the correct order, `log_decision` lines appear with non-zero tokens for LLM agents,
`PolicyAgent` produces no token lines (it makes no LLM calls), final output includes `total_tokens`
reflecting cumulative cost from the three LLM agents only.