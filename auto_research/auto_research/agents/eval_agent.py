# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eval Agent.

Runs the updated Planner through the Autoscaling Arena "according to the
experiment config". On success it sends an Experiment Report to the Research
Lead; if the Arena surfaces an execution error, it reports that back to the SWE
Agent instead.
"""

from __future__ import annotations

from typing import Optional, Union

from ..config import Settings
from ..environment import ArenaAdapter, PlannerUnderTest
from ..schemas import (
    EvalRequest,
    ExecutionError,
    ExperimentConfig,
    ExperimentReport,
)
from .base import run_react_agent, summarize_metrics, verdict_for

_SYSTEM = (
    "You are an Evaluation agent in an autonomous research loop on the Dynamo "
    "Planner. Evaluate the current Planner in the Autoscaling Arena by calling the "
    "tools. Call run_arena_eval with the requested mode; for a focused run also call "
    "get_experiment_logs. Then write a short, factual analysis of the scorecard — do "
    "not invent numbers beyond what the tools returned."
)


class EvalAgent:
    def __init__(self, settings: Settings, llm=None):
        self.settings = settings
        self.llm = llm

    def evaluate(
        self,
        request: EvalRequest,
        planner: PlannerUnderTest,
        arena: ArenaAdapter,
        config: ExperimentConfig,
        baseline: Optional[ExperimentReport],
    ) -> Union[ExperimentReport, ExecutionError]:
        capture: dict = {}
        analysis = ""

        if self.settings.mode == "llm":
            analysis = self._run_llm(request, planner, arena, config, capture)
            result = capture.get("result")
            if result is None:  # model never called the tool; run it ourselves
                result = arena.run(planner, config, request.mode)
        else:
            result = arena.run(planner, config, request.mode)

        # Execution error -> route back to the SWE Agent.
        if result.error:
            return ExecutionError(
                where="arena",
                message=result.error,
                traceback=result.error_traceback,
                change_title="",
            )

        metrics = result.metrics
        if baseline is None:
            verdict = "baseline"
            delta = 0.0
        else:
            delta = round(metrics.score - baseline.metrics.score, 4)
            verdict = verdict_for(delta)

        if not analysis:
            analysis = self._heuristic_analysis(metrics, verdict, delta, baseline)

        return ExperimentReport(
            iteration=request.iteration,
            mode=request.mode,
            metrics=metrics,
            verdict=verdict,
            vs_baseline_score_delta=delta,
            analysis=analysis,
            logs=result.logs,
        )

    def _run_llm(self, request, planner, arena, config, capture) -> str:
        from ..tools import make_eval_tools

        tools = make_eval_tools(planner, arena, config, capture)
        user = (
            f"Run a {request.mode.value} evaluation of the current Planner. "
            f"Reason for this eval: {request.reason or 'routine check'}. "
            "Report the scorecard and a two-sentence analysis."
        )
        return run_react_agent(
            self.llm, tools, _SYSTEM, user, self.settings.max_tool_iterations
        )

    @staticmethod
    def _heuristic_analysis(metrics, verdict, delta, baseline) -> str:
        head = summarize_metrics(metrics)
        if verdict == "baseline":
            return f"Baseline established. {head}."
        direction = (
            f"{verdict} vs baseline (Δscore={delta:+.3f})"
            if baseline is not None
            else verdict
        )
        driver = (
            "fewer SLA violations from earlier provisioning"
            if delta > 0
            else "more violations or higher GPU cost"
            if delta < 0
            else "no material change"
        )
        return f"{direction}: {driver}. {head}."
