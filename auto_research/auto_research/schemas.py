# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed messages exchanged between the agents.

Every arrow in the reference diagram is one of these payloads. Keeping them as
Pydantic models means the same objects flow through the LangGraph state, the
LLM structured-output calls, and the deterministic heuristic agents unchanged.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class EvalMode(str, Enum):
    """Granularity the Research Lead asks the Eval Agent for."""

    FULL = "full"  # full sweep across the whole workload trace
    FOCUSED = "focused"  # short slice + verbose logs, for debugging a hypothesis


class LeadAction(str, Enum):
    """What the Research Lead decides to do on a given turn."""

    REQUEST_EVAL = "request_eval"  # -> Eval Agent (e.g. establish a baseline)
    REQUEST_CHANGE = "request_change"  # -> SWE Agent
    FINALIZE = "finalize"  # -> emit PR + Research Report, stop


class ExperimentConfig(BaseModel):
    """The experiment the Autoscaling Arena runs the Planner against.

    This is the ``experiment config`` referenced by the arena arrow in the
    diagram: it pins the workload and the SLA targets that define "better".
    """

    name: str = "planner-autoscaling-study"
    objective: str = (
        "Maximize SLA attainment (TTFT + ITL) while keeping GPU replica-steps low "
        "under a bursty diurnal workload."
    )
    ttft_slo_ms: float = 250.0
    itl_slo_ms: float = 25.0
    workload: str = "diurnal_with_bursts"
    horizon_steps: int = 288
    per_replica_rps: float = 20.0
    max_replicas: int = 32
    provision_step: int = 2  # max replicas added/removed per control step (ramp lag)
    base_ttft_ms: float = 50.0
    base_itl_ms: float = 8.0
    seed: int = 7
    target_score: float = 0.87
    max_iterations: int = 4


class ChangeRequest(BaseModel):
    """Research Lead -> SWE Agent: "Request change"."""

    title: str
    hypothesis: str
    instructions: str
    strategy: str = "reactive_threshold"  # tag the heuristic SWE maps to a policy
    iteration: int = 0


class CodeChange(BaseModel):
    """SWE Agent's record of what it implemented into the Planner."""

    summary: str
    files_changed: List[str] = Field(default_factory=list)
    strategy: str = ""
    diff_preview: str = ""
    ok: bool = True


class ExecutionError(BaseModel):
    """"Report Any Execution Errors": surfaced by the Arena, routed to the SWE Agent."""

    where: str  # "arena" | "swe"
    message: str
    traceback: str = ""
    change_title: str = ""


class EvalRequest(BaseModel):
    """Research Lead -> Eval Agent: "Request full eval or focused test + logs"."""

    mode: EvalMode = EvalMode.FULL
    reason: str = ""
    want_logs: bool = False
    iteration: int = 0


class ArenaMetrics(BaseModel):
    """Raw scorecard the Autoscaling Arena returns for one Planner build."""

    sla_attainment: float
    p95_ttft_ms: float
    p95_itl_ms: float
    ttft_violations: int
    itl_violations: int
    gpu_replica_steps: float
    scaling_actions: int
    mean_replicas: float
    score: float


class ExperimentReport(BaseModel):
    """Eval Agent -> Research Lead: "Send Experiment Report"."""

    iteration: int
    mode: EvalMode
    metrics: ArenaMetrics
    verdict: str  # "baseline" | "improved" | "regressed" | "neutral"
    vs_baseline_score_delta: float = 0.0
    analysis: str = ""
    logs: str = ""


class LeadDecision(BaseModel):
    """Research Lead's turn output; drives the orchestration router."""

    action: LeadAction
    rationale: str = ""
    change_request: Optional[ChangeRequest] = None
    eval_request: Optional[EvalRequest] = None


class ResearchReport(BaseModel):
    """"Send PR, Research Report to team": the terminal artifact."""

    title: str
    summary: str
    baseline_score: float
    best_score: float
    iterations: int
    accepted_change: Optional[str] = None
    pr_title: str = ""
    pr_body: str = ""
    findings: List[str] = Field(default_factory=list)
