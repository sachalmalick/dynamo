# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Orchestration: the three agents wired into the loop from the diagram.

    research_lead --request change--> swe --implement--> [Planner]
          ^   ^                                              |
          |   |                                       arena evaluates
     report|   |exec error                                  v
          |   +--------------------- swe <--exec error-- eval
          +----------------- report (to lead) ------------+

The graph is built with LangGraph when it is installed. A dependency-free
fallback driver runs the *same* node functions and routers, so the heuristic
backend and the test suite work with nothing but the standard library. Keeping
one set of node functions means the two drivers can never drift.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Callable, Dict, List, Optional

from .agents import EvalAgent, ResearchLeadAgent, SWEAgent
from .base_state import END, LoopState
from .config import Settings
from .environment import MockAutoscalingArena, PlannerUnderTest
from .environment.planner import POLICY_FILENAME
from .schemas import (
    EvalMode,
    EvalRequest,
    ExecutionError,
    ExperimentConfig,
    ExperimentReport,
    LeadAction,
    ResearchReport,
)


class ResearchLoop:
    """Constructs the environment + agents and drives the multi-agent loop."""

    def __init__(
        self,
        config: ExperimentConfig,
        settings: Settings,
        arena=None,
        planner: Optional[PlannerUnderTest] = None,
        emit: Optional[Callable[[str, str, str], None]] = None,
    ):
        self.config = config
        self.settings = settings
        self.arena = arena or MockAutoscalingArena()
        workspace = settings.workspace_dir or os.path.join(
            tempfile.gettempdir(), "auto_research_workspace"
        )
        self.planner = planner or PlannerUnderTest(workspace)
        self.planner.reset_to_seed()  # each study starts from the seed policy

        llm = None
        if settings.mode == "llm":
            from .llm import make_llm

            llm = make_llm(settings)
        self.lead = ResearchLeadAgent(settings, llm)
        self.swe = SWEAgent(settings, llm)
        self.eval = EvalAgent(settings, llm)
        self._emit = emit or (lambda actor, arrow, text: None)

    # -- helpers ----------------------------------------------------------
    def _log(self, state: dict, actor: str, summary: str, **extra) -> None:
        state.setdefault("history", []).append(
            {"actor": actor, "summary": summary, **extra}
        )

    def initial_state(self) -> dict:
        return {
            "config": self.config,
            "iteration": 0,
            "backlog_index": 0,
            "baseline": None,
            "best": None,
            "best_strategy": "reactive_threshold",
            "latest_report": None,
            "pending_change": None,
            "pending_eval": None,
            "pending_error": None,
            "last_change": None,
            "last_strategy": "reactive_threshold",
            "fix_attempts": 0,
            "history": [],
            "final": None,
        }

    # -- nodes ------------------------------------------------------------
    def research_lead_node(self, state: dict) -> dict:
        decision = self.lead.decide(state)
        updates: Dict[str, Any] = {"decision": decision}

        if decision.action == LeadAction.REQUEST_CHANGE:
            cr = decision.change_request
            self._emit("Research Lead", "--request change-->", f"SWE Agent: {cr.title}")
            updates["pending_change"] = cr
            updates["pending_eval"] = None
            updates["pending_error"] = None
            updates["iteration"] = state.get("iteration", 0) + 1
            updates["backlog_index"] = state.get("backlog_index", 0) + 1
            self._log(state, "lead", f"requested change: {cr.strategy}")
        elif decision.action == LeadAction.REQUEST_EVAL:
            er = decision.eval_request
            self._emit(
                "Research Lead",
                "--request full eval / focused test-->",
                f"Eval Agent: {er.reason}",
            )
            updates["pending_eval"] = er
            self._log(state, "lead", f"requested {er.mode.value} eval")
        else:  # FINALIZE
            report = self.lead.finalize(state)
            self._emit(
                "Research Lead",
                "--send PR + Research Report-->",
                f"team: {report.pr_title}",
            )
            updates["final"] = report
            self._log(state, "lead", "finalized study")
        updates["history"] = state["history"]
        return updates

    def swe_node(self, state: dict) -> dict:
        updates: Dict[str, Any] = {}
        error: Optional[ExecutionError] = state.get("pending_error")
        if error is not None:
            change = self.swe.fix(error, self.planner)
            updates["fix_attempts"] = state.get("fix_attempts", 0) + 1
            updates["pending_error"] = None
            self._emit("SWE Agent", "--fix + implement-->", f"Planner: {change.summary[:80]}")
            self._log(state, "swe", f"fixed execution error -> {change.strategy}")
        else:
            change = self.swe.implement(state["pending_change"], self.planner)
            self._emit(
                "SWE Agent", "--implement change-->", f"Planner ({POLICY_FILENAME})"
            )
            self._log(state, "swe", change.summary)
        updates["last_change"] = change
        updates["last_strategy"] = change.strategy
        updates["pending_change"] = None
        # the Arena now evaluates the updated Planner
        updates["pending_eval"] = EvalRequest(
            mode=EvalMode.FULL,
            reason=f"Evaluate updated Planner ({change.strategy})",
            iteration=state.get("iteration", 0),
        )
        updates["history"] = state["history"]
        return updates

    def eval_node(self, state: dict) -> dict:
        request = state.get("pending_eval") or EvalRequest(
            mode=EvalMode.FULL, iteration=state.get("iteration", 0)
        )
        self._emit(
            "Autoscaling Arena",
            "--evaluate per experiment config-->",
            f"Eval Agent ({request.mode.value})",
        )
        outcome = self.eval.evaluate(
            request, self.planner, self.arena, self.config, state.get("baseline")
        )
        updates: Dict[str, Any] = {"pending_eval": None}

        if isinstance(outcome, ExecutionError):
            self._emit(
                "Eval Agent", "--report execution error-->", f"SWE Agent: {outcome.message[:80]}"
            )
            updates["pending_error"] = outcome
            self._log(state, "eval", f"execution error: {outcome.message[:60]}")
            updates["history"] = state["history"]
            return updates

        report: ExperimentReport = outcome
        strategy = state.get("last_strategy", "reactive_threshold")
        self._emit(
            "Eval Agent",
            "--send experiment report-->",
            f"Research Lead: {report.verdict} score={report.metrics.score:.3f}",
        )
        updates["latest_report"] = report
        if state.get("baseline") is None:
            updates["baseline"] = report
            strategy = "reactive_threshold (seed)"
        best = state.get("best")
        if best is None or report.metrics.score > best.metrics.score:
            updates["best"] = report
            updates["best_strategy"] = state.get("last_strategy", "reactive_threshold")
        updates["fix_attempts"] = 0
        self._log(
            state,
            "eval",
            f"{report.verdict}: {report.analysis[:80]}",
            score=report.metrics.score,
            strategy=strategy,
            iteration=report.iteration,
        )
        updates["history"] = state["history"]
        return updates

    # -- routers ----------------------------------------------------------
    @staticmethod
    def route_from_lead(state: dict) -> str:
        action = state["decision"].action
        if action == LeadAction.REQUEST_CHANGE:
            return "swe"
        if action == LeadAction.REQUEST_EVAL:
            return "eval"
        return END

    @staticmethod
    def route_from_eval(state: dict) -> str:
        return "swe" if state.get("pending_error") is not None else "research_lead"

    # -- drivers ----------------------------------------------------------
    def build_langgraph(self):
        """Compile the LangGraph StateGraph, or return None if unavailable."""
        try:
            from langgraph.graph import END as LG_END
            from langgraph.graph import START, StateGraph
        except ImportError:
            return None

        graph = StateGraph(LoopState)
        graph.add_node("research_lead", self.research_lead_node)
        graph.add_node("swe", self.swe_node)
        graph.add_node("eval", self.eval_node)
        graph.add_edge(START, "research_lead")
        graph.add_conditional_edges(
            "research_lead",
            self.route_from_lead,
            {"swe": "swe", "eval": "eval", END: LG_END},
        )
        graph.add_edge("swe", "eval")
        graph.add_conditional_edges(
            "eval",
            self.route_from_eval,
            {"swe": "swe", "research_lead": "research_lead"},
        )
        return graph.compile()

    def _run_fallback(self, state: dict, max_steps: int) -> dict:
        """Dependency-free driver of the same nodes + routers."""
        node = "research_lead"
        steps = 0
        while node != END and steps < max_steps:
            steps += 1
            if node == "research_lead":
                state.update(self.research_lead_node(state))
                node = self.route_from_lead(state)
            elif node == "swe":
                state.update(self.swe_node(state))
                node = "eval"
            elif node == "eval":
                state.update(self.eval_node(state))
                node = self.route_from_eval(state)
            else:  # pragma: no cover - defensive
                break
        return state

    def run(self, use_langgraph: bool = True, max_steps: int = 60) -> ResearchReport:
        state = self.initial_state()
        compiled = self.build_langgraph() if use_langgraph else None
        if compiled is not None:
            state = dict(
                compiled.invoke(state, config={"recursion_limit": max_steps})
            )
        else:
            state = self._run_fallback(state, max_steps)
        return state["final"]
