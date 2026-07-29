# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eval Agent tools: run the Autoscaling Arena and pull logs.

The arena tool stashes its authoritative ``ArenaResult`` in the shared
``capture`` dict. The graph node reads that — not the model's summary — to build
the ExperimentReport and to decide whether an execution error must be routed
back to the SWE Agent.
"""

from __future__ import annotations

import json
from typing import Dict, List

from langchain_core.tools import tool

from ..environment import ArenaAdapter, PlannerUnderTest
from ..schemas import EvalMode, ExperimentConfig


def make_eval_tools(
    planner: PlannerUnderTest,
    arena: ArenaAdapter,
    config: ExperimentConfig,
    capture: Dict,
) -> List:
    """Build arena/log tools bound to a Planner, Arena, and experiment config."""

    @tool
    def run_arena_eval(mode: str = "full") -> str:
        """Evaluate the current Planner in the Autoscaling Arena.

        ``mode`` is "full" (whole workload) or "focused" (short slice with
        verbose per-step logs). Returns a JSON scorecard, or an error string if
        the Planner failed to load or raised during the run.
        """
        eval_mode = EvalMode.FOCUSED if mode == "focused" else EvalMode.FULL
        result = arena.run(planner, config, eval_mode)
        capture["result"] = result
        capture["mode"] = eval_mode
        if result.error:
            return f"EXECUTION_ERROR: {result.error}"
        return json.dumps(result.metrics.model_dump(), indent=2)

    @tool
    def get_experiment_logs() -> str:
        """Return per-step logs from the most recent focused arena run."""
        result = capture.get("result")
        if result is None:
            return "no evaluation has been run yet"
        return result.logs or "no violating steps were logged"

    return [run_arena_eval, get_experiment_logs]
