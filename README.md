# Sokoban solver

A push-optimal Sokoban solver for make.fun's "Build the best Sokoban solver,
21 November 2026" bounty.

## How it works

The levels are a hallway with one to four rooms off it, each room holding 3
or 4 boxes and as many goals. A box pushed out of a door into the hallway can
never be pushed back, so every optimal solution keeps each box in its room
and the level's push-optimal count is the sum of its rooms' optima. The
solver therefore splits a level into rooms, solves each room on its own with
a push-space A* search (Hungarian assignment of boxes to goals as the
estimate, dead-square and frozen-box pruning, the player's reachable region
folded into each state's key), and walks the player between rooms through
the hallway. Greedy best-first and breadth-first passes of the same search
stand by if the A* pass ever runs out of its state budget. Every search is
budgeted by work, never by the clock, so two runs write the same bytes.

## Run

```sh
node main.ts
```

It reads `/input/<id>/level.txt` for every instance and writes
`/output/<id>/solution.txt` (the player's moves in LURD, lower case a step,
upper case a push). `MAKEFUN_INPUT_DIR` and `MAKEFUN_OUTPUT_DIR` stand in
for `/input` and `/output` outside a container.
