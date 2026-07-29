# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWE Agent.

Receives a "Request change" from the Research Lead and implements it into the
Planner (Execution Environment). Also receives "Report Any Execution Errors"
routed from the Autoscaling Arena via the Eval Agent, and fixes the broken build.
"""

from __future__ import annotations

from ..config import Settings
from ..environment import PlannerUnderTest
from ..environment.planner import POLICY_FILENAME
from ..environment.strategies import STRATEGIES
from ..schemas import ChangeRequest, CodeChange, ExecutionError
from .base import run_react_agent

_SYSTEM = (
    "You are a Software Engineer agent working inside an autonomous research loop "
    "on the Dynamo Planner (an SLA-driven autoscaling controller). The Planner "
    "policy is a single Python module exposing `decide(ctx) -> int`, which returns "
    "the target replica count for the next control step. Use the tools to read the "
    "current source and write an improved version. Keep the `decide(ctx)` contract "
    "and the SPDX header. Never fabricate results — only edit code. `ctx` exposes: "
    "step, current_replicas, offered_rps, recent_rps (list, newest last), "
    "utilization, p_ttft_ms, p_itl_ms, per_replica_rps, max_replicas, ttft_slo_ms, "
    "itl_slo_ms."
)


class SWEAgent:
    def __init__(self, settings: Settings, llm=None):
        self.settings = settings
        self.llm = llm

    # -- Request change ---------------------------------------------------
    def implement(self, change: ChangeRequest, planner: PlannerUnderTest) -> CodeChange:
        if self.settings.mode == "llm":
            return self._implement_llm(change, planner)
        return self._implement_heuristic(change, planner)

    def _implement_heuristic(
        self, change: ChangeRequest, planner: PlannerUnderTest
    ) -> CodeChange:
        source = STRATEGIES.get(change.strategy)
        if source is None:
            source = planner.read_source()
            summary = (
                f"No policy registered for strategy '{change.strategy}'; left the "
                "Planner unchanged."
            )
            ok = False
        else:
            planner.write_source(source)
            summary = f"Implemented the '{change.strategy}' policy for: {change.title}"
            ok = True
        return CodeChange(
            summary=summary,
            files_changed=[POLICY_FILENAME],
            strategy=change.strategy,
            diff_preview=_head(planner.read_source()),
            ok=ok,
        )

    def _implement_llm(
        self, change: ChangeRequest, planner: PlannerUnderTest
    ) -> CodeChange:
        from ..tools import make_swe_tools

        tools = make_swe_tools(planner)
        user = (
            f"Change request: {change.title}\n"
            f"Hypothesis: {change.hypothesis}\n\n"
            f"Implement this: {change.instructions}\n\n"
            "First call read_planner_source, then call write_planner_source with the "
            "complete new module. Reply with a one-paragraph summary of what you changed."
        )
        summary = run_react_agent(
            self.llm, tools, _SYSTEM, user, self.settings.max_tool_iterations
        )
        return CodeChange(
            summary=summary or f"Applied change: {change.title}",
            files_changed=[POLICY_FILENAME],
            strategy=change.strategy,
            diff_preview=_head(planner.read_source()),
            ok=True,
        )

    # -- Report Any Execution Errors (fix path) ---------------------------
    def fix(self, error: ExecutionError, planner: PlannerUnderTest) -> CodeChange:
        if self.settings.mode == "llm":
            return self._fix_llm(error, planner)
        return self._fix_heuristic(error, planner)

    def _fix_heuristic(
        self, error: ExecutionError, planner: PlannerUnderTest
    ) -> CodeChange:
        # The heuristic SWE never emits broken code; if the arena still reported an
        # error, fall back to a known-good reactive policy.
        planner.write_source(STRATEGIES["reactive_threshold"])
        return CodeChange(
            summary=(
                "Reverted the Planner to the known-good reactive_threshold policy "
                f"after an execution error: {error.message}"
            ),
            files_changed=[POLICY_FILENAME],
            strategy="reactive_threshold",
            diff_preview=_head(planner.read_source()),
            ok=True,
        )

    def _fix_llm(self, error: ExecutionError, planner: PlannerUnderTest) -> CodeChange:
        from ..tools import make_swe_tools

        tools = make_swe_tools(planner)
        user = (
            "The Autoscaling Arena reported an execution error while evaluating your "
            f"last Planner edit:\n\n{error.message}\n\n{error.traceback}\n\n"
            "Read the current source, fix the bug, and write the corrected module. "
            "Reply with a one-line summary of the fix."
        )
        summary = run_react_agent(
            self.llm, tools, _SYSTEM, user, self.settings.max_tool_iterations
        )
        return CodeChange(
            summary=summary or "Fixed the execution error in the Planner policy.",
            files_changed=[POLICY_FILENAME],
            strategy="fix",
            diff_preview=_head(planner.read_source()),
            ok=True,
        )


def _head(source: str, n: int = 16) -> str:
    lines = source.splitlines()
    body = [ln for ln in lines if not ln.startswith("#") and ln.strip()]
    return "\n".join(body[:n])
