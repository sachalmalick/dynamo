<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Auto Research

A LangChain / LangGraph **multi-agent framework** that autonomously improves the
Dynamo **Planner** (the SLA-driven autoscaling controller) by looping code
changes through an **Autoscaling Arena** and letting a **Research Lead** decide
when to ship.

It implements the reference architecture directly:

```
                          Research Lead Agent ───── send PR + Research Report ──▶ team
                            │        ▲   ▲
              request change│  report│   │request full eval / focused test + logs
                            ▼        │   ▼
                        SWE Agent    │   Eval Agent
                            │        │      ▲
                   implement│        │      │ evaluate updated Planner
                     change ▼        │      │  per experiment config
        ┌─ Execution Environment ────┼──────┼────────────────┐
        │        Planner  ◀───────── report execution error  │
        │           ▲                       │                │
        │           └──── Autoscaling Arena ─┘               │
        └───────────────────────────────────────────────────┘
```

- **Research Lead Agent** — owns the loop. Requests changes from the SWE Agent,
  requests full or focused evals from the Eval Agent, reads Experiment Reports,
  and emits the final PR + Research Report.
- **SWE Agent** — implements a requested change into the Planner policy, and
  fixes execution errors reported back from the Arena.
- **Eval Agent** — runs the updated Planner through the Autoscaling Arena
  "according to the experiment config"; sends an Experiment Report on success, or
  routes an execution error back to the SWE Agent.
- **Execution Environment** — the **Planner** (an editable policy module) and the
  **Autoscaling Arena** (the eval harness).

## The Autoscaling Arena is stubbed

The real Autoscaling Arena is not available here, so `MockAutoscalingArena`
stands in for it: a deterministic, seeded simulator that drives the current
Planner policy against a bursty diurnal workload and returns an SLA/cost
scorecard. It is a drop-in behind the `ArenaAdapter` protocol
(`environment/autoscaling_arena.py`) — replace it with a real adapter and nothing
else in the framework changes.

Because provisioning is rate-limited (decode replicas take time to spin up), a
policy that only reacts after load arrives lags bursts and violates SLAs, while
one that forecasts load and pre-provisions rides them out. That tension, plus a
GPU-cost term, is what the score rewards — so code changes produce real,
measurable movement.

## Two backends, one graph

| Mode | Agents are… | Needs |
|------|-------------|-------|
| `heuristic` (default offline) | deterministic stand-ins | just `pydantic` |
| `llm` | LangChain tool-calling / structured-output agents | `langchain-*`, `langgraph`, an API key |

Both drive the **same** LangGraph `StateGraph` over the same environment and
tools, so the wiring is identical either way. A dependency-free fallback driver
runs the same nodes and routers when LangGraph is not installed.

## Quickstart

```bash
cd auto_research
pip install -r requirements.txt          # or just `pip install pydantic` for offline

# Offline, deterministic (heuristic agents + fallback driver):
python -m auto_research --no-langgraph

# Real LangChain agents:
export ANTHROPIC_API_KEY=sk-ant-...
export AUTO_RESEARCH_MODE=llm
python -m auto_research
```

Programmatic use:

```python
from auto_research import ExperimentConfig, ResearchLoop, Settings

loop = ResearchLoop(ExperimentConfig(), Settings.from_env())
report = loop.run()
print(report.pr_title, report.best_score)
print(loop.planner.read_source())   # the winning Planner policy
```

See `examples/improve_planner.py` for a runnable walkthrough.

## Layout

```
auto_research/
├── schemas.py            # typed messages = the arrows in the diagram
├── config.py             # Settings (mode / provider / model)
├── llm.py                # LangChain chat-model factory
├── graph.py              # ResearchLoop: nodes, routers, LangGraph + fallback driver
├── base_state.py         # LoopState TypedDict
├── agents/               # research_lead, swe_agent, eval_agent (llm + heuristic)
├── tools/                # LangChain tools over the environment (swe / eval)
├── environment/
│   ├── planner.py            # PlannerUnderTest — the editable policy
│   ├── seed_policy.py        # seed reactive policy the SWE Agent edits
│   ├── strategies.py         # policy library for the heuristic SWE
│   └── autoscaling_arena.py  # ArenaAdapter + MockAutoscalingArena
├── examples/improve_planner.py
└── tests/test_loop.py    # runs with only pydantic (no LLM, no key)
```

## Configuration

Env vars (all optional; see `.env.example`):

| Var | Default | Meaning |
|-----|---------|---------|
| `AUTO_RESEARCH_MODE` | auto | `llm` or `heuristic` |
| `AUTO_RESEARCH_PROVIDER` | `anthropic` | `anthropic` or `openai` |
| `AUTO_RESEARCH_MODEL` | `claude-sonnet-5` | chat model id |
| `AUTO_RESEARCH_WORKSPACE` | temp dir | where the editable Planner lives |

CLI flags mirror these: `--mode`, `--provider`, `--model`, `--workload`,
`--horizon`, `--max-iterations`, `--target-score`, `--no-langgraph`, `--quiet`.

## Tests

```bash
cd auto_research
python tests/test_loop.py          # no pytest needed
# or
pytest tests/                      # standard pytest
```

## Extending

- **Real Arena** — implement `ArenaAdapter.run(planner, config, mode) -> ArenaResult`
  and pass it to `ResearchLoop(arena=...)`.
- **New workloads / SLAs** — fields on `ExperimentConfig`.
- **New candidate policies** — add to `environment/strategies.py` (heuristic) or
  let the LLM SWE Agent generate them.
- **Different objective** — change the `score` formula in `autoscaling_arena.py`.
