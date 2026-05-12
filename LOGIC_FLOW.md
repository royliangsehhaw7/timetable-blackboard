# Scheduler.run() — Data Flow Simulation
Setup Two courses seeded on the board at start:
- CS201 — Data Structures, requires lab
- MA101 — Calculus, no lab required

> [!IMPORTANT]
> Agents in poll order: CourseAgent → RoomAgent → LecturerAgent → PolicyAgent

## CYCLE 1
Blackboard state entering cycle
```js
unscheduled : [CS201, MA101]
proposals   : {}
assignments : []
```
**Lifecycle pass** 
- board.is_complete() → False. Continue.
- Lifecycle pass — no proposals in flight, nothing to process.
- board.ensure_proposals_exist()
- CS201 has no in-flight proposal. One created.
```js
proposals : { P1 → { course: CS201, timeslot: None, room: None, lecturer: None } }
```

**Agent poll**
- CourseAgent.is_competent_for() — finds P1, timeslot is None → **MATCH**
- Activates. Reads confirmed assignments (none) and timeslot conflicts (none).
- Proposes Monday 10–11.
- Writes P1 back to board.

Blackboard state leaving cycle
```js
proposals : { P1 → { course: CS201, timeslot: Mon 10-11, room: None, lecturer: None } }
```

## CYCLE 2
*Blackboard state entering cycle*
```js
unscheduled : [CS201, MA101]
proposals   : { P1 → { course: CS201, timeslot: Mon 10-11, room: None, lecturer: None } }
assignments : []
```
**Lifecycle pass** 
- P1 is not confirmed, rejected, or exhausted. Nothing to process.
- board.ensure_proposals_exist()
- MA101 has no in-flight proposal. One created.

```js
proposals : {
  P1 → { course: CS201, timeslot: Mon 10-11, room: None, lecturer: None },
  P2 → { course: MA101, timeslot: None,      room: None, lecturer: None }
}
```
**Agent poll**
- CourseAgent.is_competent_for() — finds P2, timeslot is None → **MATCH**
- Reads confirmed assignments (none).
- Reads **board.get_timeslot_conflicts(P2)** — P1 has Mon 10–11, so that slot is reported as claimed.
- Proposes Tuesday 09–10 (avoids Monday 10–11).
- Writes P2 back to board.

*Blackboard state leaving cycle*
```js
proposals : {
  P1 → { course: CS201, timeslot: Mon 10-11, room: None, lecturer: None },
  P2 → { course: MA101, timeslot: Tue 09-10, room: None, lecturer: None }
}
```
> [!NOTE]
> Conflict detection in action — CourseAgent saw P1's timeslot on the board and avoided it before proposing. No clash needed PolicyAgent to catch.

## CYCLE 3
**Lifecycle pass** 
- nothing to process.
- board.ensure_proposals_exist() — both courses already have proposals. Nothing created.

**Agent poll**
- CourseAgent.is_competent_for() — P1 timeslot set, P2 timeslot set. No match.
- RoomAgent.is_competent_for() — finds P1: timeslot set, room None → **MATCH**
- Reads confirmed rooms at Mon 10–11 (none).
- Reads **board.get_room_conflicts(P1)** — P2 is at a different timeslot, no conflict.
- CS201 requires lab. Assigns R003 — Computer Lab A.
- Writes P1 back.

*Blackboard state leaving cycle*
```js
proposals : {
  P1 → { course: CS201, timeslot: Mon 10-11, room: R003, lecturer: None },
  P2 → { course: MA101, timeslot: Tue 09-10, room: None, lecturer: None }
}
```

## CYCLE 4
**Agent poll**
- CourseAgent — no match.
- RoomAgent.is_competent_for() — finds P2: timeslot set, room None → MATCH
- MA101 does not require lab. Reads rooms at Tue 09–10.
- Reads **board.get_room_conflicts(P2)** — P1 is at Mon 10–11, different slot, no conflict.
- Assigns R001 — Lecture Theatre 1.
- Writes P2 back.

*Blackboard state leaving cycle*
```js
proposals : {
  P1 → { course: CS201, timeslot: Mon 10-11, room: R003, lecturer: None },
  P2 → { course: MA101, timeslot: Tue 09-10, room: R001, lecturer: None }
}
```

## CYCLE 5

**Agent poll**
- CourseAgent — no match.
- RoomAgent — no match (both have rooms).
- LecturerAgent.is_competent_for() — finds P1: room set, lecturer None → MATCH
- CS201 needs a lecturer who teaches it. Reads confirmed lecturers at Mon 10–11 (none).
- Reads **board.get_lecturer_conflicts(P1)** — P2 at Tue 09–10, no conflict.
- Assigns L001 — Dr. Okafor (teaches CS201, available Mon 10–11).
- Writes P1 back.

*Blackboard state leaving cycle*
```js
proposals : {
  P1 → { course: CS201, timeslot: Mon 10-11, room: R003, lecturer: L001 },
  P2 → { course: MA101, timeslot: Tue 09-10, room: R001, lecturer: None }
}
```

## CYCLE 6

**Agent poll**
- CourseAgent, RoomAgent — no match.
- LecturerAgent.is_competent_for() — finds P2: room set, lecturer None → MATCH
- MA101 needs a qualified lecturer at Tue 09–10.
- Reads **board.get_lecturer_conflicts(P2)** — P1 has L001 at Mon 10–11, different slot, no conflict.
- Assigns L002 — Prof. Singh (teaches MA101, no unavailability).
- Writes P2 back.

*Blackboard state leaving cycle*
```js
proposals : {
  P1 → { course: CS201, timeslot: Mon 10-11, room: R003, lecturer: L001, policy_approved: None },
  P2 → { course: MA101, timeslot: Tue 09-10, room: R001, lecturer: L002, policy_approved: None }
}
```

## CYCLE 7
**Agent poll**
- CourseAgent, RoomAgent, LecturerAgent — no match.
- PolicyAgent.is_competent_for() — finds P1: lecturer set, policy_approved None → - **MATCH**
- Runs check(P1):
  - Mon is a valid school day ✓
  - 10–11 within school hours ✓
  - No lunch overlap ✓
  - CS201 requires lab, R003 is a lab ✓
  - L001 teaches CS201 ✓
  - L001 not unavailable Mon 10–11 ✓
  - **get_room_conflicts(P1)** → none ✓
  - **get_lecturer_conflicts(P1)** → non ✓
  - Sets policy_approved = True. Writes P1 back.

*Blackboard state leaving cycle*
```js
proposals : {
  P1 → { course: CS201, ..., policy_approved: True },
  P2 → { course: MA101, ..., policy_approved: None }
}
```

