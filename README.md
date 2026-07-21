# REFUGIO Madrid 2026 — Warehouse Challenge Local Kit

This is the challenge engine and local evaluator from the REFUGIO Madrid 2026
hackathon (July 4, 2026), released alongside the whitepaper *Speedrunning as a
model of progress*. It lets anyone run the exact task that both the human
teams and the autonomous agents competed on, and reproduce official scores on
their own machine.

The challenge: a decentralized multi-agent warehouse routing problem. You
submit a single Python file exposing an `act(observation)` policy (and
optionally a `create_layout()` warehouse layout). The full rules are in
`WAREHOUSE_CHALLENGE.md`.

## Setup

Requires Python 3.11+.

```bash
python -m pip install -e ".[dev]"
```

The engine itself has no runtime dependencies. During the event, submissions
were additionally allowed to import `numpy`, `scipy`, `networkx`,
`sortedcontainers`, and `numba`; install those if the policy you are testing
uses them.

## Run a policy locally

```bash
python -m warehouse.local_runner examples/baseline_submission.py
```

This uses the same engine, tick count (300), and per-evaluation policy compute
budget (180 s) that the official server enforced, but with placeholder local
seeds.

## Reproduce official scores

During the event, scores were computed as total deliveries summed across three
hidden evaluation seeds. The event is over, so the seeds are now public; they
are listed in `EVAL_SEEDS.txt`. To evaluate exactly as the official server
did:

```bash
python -m warehouse.local_runner path/to/submission.py \
  --seeds bff0fb14575b4676b1f0f01bfc7b0126,dfbf918495ee4fca8d50b53456d59fa8,546a597410b049de82f7ce72fe7fd714
```

The printed `score` field matches the official leaderboard value for that
submission. (Timing-related fields naturally vary with hardware; the score
itself is deterministic unless the policy exhausts the compute budget.)

Note that the whitepaper describes a late-event phase in which human teams
fine-tuned policies against these exact seeds after they leaked. With the
seeds public, any new score should be read as fixed-instance optimization, not
as general routing performance.

## Tests

```bash
python -m pytest
```

## Run data

The full record of the event and the autonomous-agent runs — all submitted
policies, per-submission scores and timestamps, and replay files — is
available on request from the author (@XMihura).
