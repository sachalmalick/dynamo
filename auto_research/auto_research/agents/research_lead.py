# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Research Lead Agent.

Owns the loop. Requests changes from the SWE Agent, requests full or focused
evals from the Eval Agent, reads Experiment Reports, and when the study is done
emits the PR + Research Report "to the team".
"""

from __future__ import annotations

from typing import Optional

from ..config import Settings
from ..environment.strategies import STRATEGY_BACKLOG
from ..schemas import (
    ChangeRequest,
    EvalMode,
    EvalRequest,
    LeadAction,
    LeadDecision,
    ResearchReport,
)
from .base import history_digest, summarize_metrics

_SYSTEM = (
    "You are the Research Lead of an autonomous team improving the Dynamo Planner, "
    "an SLA-driven autoscaling controller. Your goal is to raise the Autoscaling "
    "Arena score (0.75*SLA_attainment + 0.25*(1 - normalized_GPU_cost)) under a "
    "bursty diurnal workload with provisioning lag. You direct a SWE Agent (which "
    "edits the Planner policy) and an Eval Agent (which runs the Arena). Decide the "
    "next action: establish a baseline eval, request a concrete code change with a "
    "testable hypothesis, or finalize once the score plateaus or the budget is spent."
)


class ResearchLeadAgent:
    def __init__(self, settings: Settings, llm=None):
        self.settings = settings
        self.llm = llm

    # -- decide the next action ------------------------------------------
    def decide(self, state: dict) -> LeadDecision:
        if self.settings.mode == "llm":
            decision = self._decide_llm(state)
            if decision is not None:
                return decision
        return self._decide_heuristic(state)

    def _decide_heuristic(self, state: dict) -> LeadDecision:
        config = state["config"]
        baseline = state.get("baseline")

        if baseline is None:
            return LeadDecision(
                action=LeadAction.REQUEST_EVAL,
                rationale="No baseline yet — measure the seed Planner first.",
                eval_request=EvalRequest(
                    mode=EvalMode.FULL,
                    reason="Establish a baseline scorecard for the seed policy.",
                    iteration=0,
                ),
            )

        best = state.get("best")
        idx = state.get("backlog_index", 0)
        reached_target = best is not None and best.metrics.score >= config.target_score
        exhausted = idx >= len(STRATEGY_BACKLOG)
        at_max = state.get("iteration", 0) >= config.max_iterations

        if reached_target or exhausted or at_max:
            reason = (
                "target score reached"
                if reached_target
                else "backlog exhausted"
                if exhausted
                else "iteration budget spent"
            )
            return LeadDecision(
                action=LeadAction.FINALIZE,
                rationale=f"Stopping: {reason}. Best score "
                f"{best.metrics.score:.3f}" if best else "Stopping.",
            )

        strategy, hypothesis, instructions = STRATEGY_BACKLOG[idx]
        return LeadDecision(
            action=LeadAction.REQUEST_CHANGE,
            rationale=f"Testing hypothesis via '{strategy}'.",
            change_request=ChangeRequest(
                title=f"Improve autoscaling with {strategy}",
                hypothesis=hypothesis,
                instructions=instructions,
                strategy=strategy,
                iteration=state.get("iteration", 0) + 1,
            ),
        )

    def _decide_llm(self, state: dict) -> Optional[LeadDecision]:
        try:
            config = state["config"]
            baseline = state.get("baseline")
            best = state.get("best")
            latest = state.get("latest_report")
            ctx = (
                f"Experiment: {config.name}\nObjective: {config.objective}\n"
                f"SLOs: TTFT<= {config.ttft_slo_ms}ms, ITL<= {config.itl_slo_ms}ms\n"
                f"target_score={config.target_score}, "
                f"iteration={state.get('iteration', 0)}/{config.max_iterations}\n"
                f"baseline: {summarize_metrics(baseline.metrics) if baseline else 'none'}\n"
                f"best: {summarize_metrics(best.metrics) if best else 'none'}\n"
                f"latest: {summarize_metrics(latest.metrics) if latest else 'none'}\n"
                f"history:\n{history_digest(state.get('history', []))}\n\n"
                "Choose the next action. If there is no baseline, request_eval. "
                "Otherwise either request_change (with a concrete hypothesis + "
                "instructions for the SWE agent) or finalize."
            )
            structured = self.llm.with_structured_output(LeadDecision)
            from langchain_core.messages import HumanMessage, SystemMessage

            return structured.invoke(
                [SystemMessage(content=_SYSTEM), HumanMessage(content=ctx)]
            )
        except Exception:
            return None  # fall back to the deterministic planner

    # -- finalize: PR + Research Report ----------------------------------
    def finalize(self, state: dict) -> ResearchReport:
        config = state["config"]
        baseline = state.get("baseline")
        best = state.get("best")
        base_score = baseline.metrics.score if baseline else 0.0
        best_score = best.metrics.score if best else base_score
        accepted = state.get("best_strategy")

        findings = []
        for ev in state.get("history", []):
            if ev.get("actor") == "eval" and ev.get("score") is not None:
                findings.append(
                    f"iter {ev.get('iteration', '?')} [{ev.get('strategy', 'seed')}]: "
                    f"score={ev['score']:.3f}"
                )

        report = self._finalize_llm(state, base_score, best_score) if (
            self.settings.mode == "llm"
        ) else None
        if report is not None:
            report.findings = findings or report.findings
            return report

        improved = best_score - base_score
        summary = (
            f"Ran an autonomous {state.get('iteration', 0)}-iteration study to improve "
            f"the Dynamo Planner's autoscaling policy under the '{config.workload}' "
            f"workload. Best policy '{accepted}' scored {best_score:.3f} vs the seed "
            f"reactive baseline {base_score:.3f} (Δ{improved:+.3f}), driven mainly by "
            "fewer SLA violations from forecasting load and pre-provisioning ahead of "
            "the provisioning lag."
        )
        pr_body = _render_pr_body(config, baseline, best, accepted, findings)
        return ResearchReport(
            title=f"Autoscaling study: {accepted or 'seed'} policy",
            summary=summary,
            baseline_score=round(base_score, 4),
            best_score=round(best_score, 4),
            iterations=state.get("iteration", 0),
            accepted_change=accepted,
            pr_title=f"feat(planner): adopt {accepted} autoscaling policy",
            pr_body=pr_body,
            findings=findings,
        )

    def _finalize_llm(self, state, base_score, best_score) -> Optional[ResearchReport]:
        try:
            config = state["config"]
            ctx = (
                f"Write the final PR + research report. Objective: {config.objective}. "
                f"Baseline score {base_score:.3f}, best score {best_score:.3f}, "
                f"accepted policy '{state.get('best_strategy')}'. History:\n"
                f"{history_digest(state.get('history', []), limit=20)}\n"
                "Fill every field. pr_title must be a Conventional Commit "
                "(feat(planner): ...)."
            )
            structured = self.llm.with_structured_output(ResearchReport)
            from langchain_core.messages import HumanMessage, SystemMessage

            return structured.invoke(
                [SystemMessage(content=_SYSTEM), HumanMessage(content=ctx)]
            )
        except Exception:
            return None


def _render_pr_body(config, baseline, best, accepted, findings) -> str:
    base_line = summarize_metrics(baseline.metrics) if baseline else "n/a"
    best_line = summarize_metrics(best.metrics) if best else "n/a"
    findings_block = "\n".join(f"- {f}" for f in findings) or "- (none)"
    return (
        "## Summary\n"
        f"Adopts the `{accepted}` autoscaling policy for the Dynamo Planner, selected "
        f"by an autonomous research loop evaluated in the Autoscaling Arena against the "
        f"`{config.workload}` workload (TTFT SLO {config.ttft_slo_ms}ms, ITL SLO "
        f"{config.itl_slo_ms}ms).\n\n"
        "## Results\n"
        f"- Baseline (seed reactive): {base_line}\n"
        f"- Accepted ({accepted}): {best_line}\n\n"
        "## Iterations\n"
        f"{findings_block}\n\n"
        "## Validation\n"
        "- Autoscaling Arena full-workload eval (deterministic, seeded).\n"
    )
