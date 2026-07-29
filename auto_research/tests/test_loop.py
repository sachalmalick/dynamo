# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Auto Research loop.

These exercise the heuristic backend and the dependency-free driver, so they run
with only pydantic installed — no LangChain, LangGraph, or API keys required.
"""

from __future__ import annotations

import os
import sys

# Make the package importable however pytest is invoked.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auto_research.agents import EvalAgent  # noqa: E402
from auto_research.config import Settings  # noqa: E402
from auto_research.environment import MockAutoscalingArena, PlannerUnderTest  # noqa: E402
from auto_research.environment.strategies import STRATEGIES  # noqa: E402
from auto_research.graph import ResearchLoop  # noqa: E402
from auto_research.schemas import EvalMode, EvalRequest, ExecutionError, ExperimentConfig  # noqa: E402


def _settings(tmp_path) -> Settings:
    return Settings(mode="heuristic", workspace_dir=str(tmp_path))


def test_seed_policy_loads_and_decides(tmp_path):
    planner = PlannerUnderTest(str(tmp_path))
    planner.reset_to_seed()
    decide = planner.load_decider()

    class Ctx:
        step = 0
        current_replicas = 4
        offered_rps = 100.0
        recent_rps = [80.0, 90.0, 100.0]
        utilization = 1.25
        p_ttft_ms = 50.0
        p_itl_ms = 8.0
        per_replica_rps = 20.0
        max_replicas = 32
        ttft_slo_ms = 250.0
        itl_slo_ms = 25.0

    target = decide(Ctx())
    assert isinstance(target, int)
    assert 1 <= target <= 32


def test_all_strategies_load_and_return_valid_replicas(tmp_path):
    planner = PlannerUnderTest(str(tmp_path))
    arena = MockAutoscalingArena()
    config = ExperimentConfig(horizon_steps=96)
    for name, source in STRATEGIES.items():
        planner.write_source(source)
        result = arena.run(planner, config, EvalMode.FULL)
        assert result.error is None, f"{name} raised: {result.error}"
        assert result.metrics is not None
        assert 0.0 <= result.metrics.sla_attainment <= 1.0
        assert 0.0 <= result.metrics.score <= 1.0
        assert result.metrics.mean_replicas >= 1.0


def test_arena_is_deterministic(tmp_path):
    planner = PlannerUnderTest(str(tmp_path))
    planner.reset_to_seed()
    arena = MockAutoscalingArena()
    config = ExperimentConfig(horizon_steps=120, seed=42)
    a = arena.run(planner, config, EvalMode.FULL)
    b = arena.run(planner, config, EvalMode.FULL)
    assert a.metrics.model_dump() == b.metrics.model_dump()


def test_arena_reports_execution_error_on_broken_policy(tmp_path):
    planner = PlannerUnderTest(str(tmp_path))
    planner.write_source("def decide(ctx):\n    return undefined_name\n")
    arena = MockAutoscalingArena()
    result = arena.run(planner, ExperimentConfig(horizon_steps=48), EvalMode.FULL)
    assert result.error is not None
    assert result.metrics is None


def test_eval_agent_routes_broken_policy_to_execution_error(tmp_path):
    planner = PlannerUnderTest(str(tmp_path))
    planner.write_source("def decide(ctx):\n    raise ValueError('boom')\n")
    agent = EvalAgent(_settings(tmp_path))
    outcome = agent.evaluate(
        EvalRequest(mode=EvalMode.FULL),
        planner,
        MockAutoscalingArena(),
        ExperimentConfig(horizon_steps=48),
        baseline=None,
    )
    assert isinstance(outcome, ExecutionError)
    assert outcome.where == "arena"


def test_full_loop_produces_report_and_improves(tmp_path):
    config = ExperimentConfig(horizon_steps=144, max_iterations=4)
    loop = ResearchLoop(config, _settings(tmp_path))
    report = loop.run(use_langgraph=False)

    assert report is not None
    assert report.iterations >= 1
    # the accepted policy must be at least as good as the seed baseline
    assert report.best_score >= report.baseline_score
    assert report.accepted_change in STRATEGIES
    assert report.pr_title.startswith("feat(planner):")
    # baseline + at least one experiment recorded
    assert len(report.findings) >= 2


def test_loop_recovers_from_execution_error(tmp_path):
    """A broken build surfaced by the Arena must route to the SWE fix path."""
    config = ExperimentConfig(horizon_steps=96, max_iterations=2)
    loop = ResearchLoop(config, _settings(tmp_path))
    state = loop.initial_state()

    # simulate an already-implemented broken change awaiting evaluation
    loop.planner.write_source("def decide(ctx):\n    return 1/0\n")
    state["pending_eval"] = EvalRequest(mode=EvalMode.FULL)
    state["baseline"] = None

    state.update(loop.eval_node(state))
    assert state["pending_error"] is not None
    assert loop.route_from_eval(state) == "swe"

    # SWE fixes it; a subsequent eval must succeed
    state.update(loop.swe_node(state))
    assert state["pending_error"] is None
    state.update(loop.eval_node(state))
    assert state["pending_error"] is None
    assert state["latest_report"] is not None


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                sub = tempfile.mkdtemp(dir=d)
                fn(sub)  # type: ignore[arg-type]
                print(f"PASS {name}")
    print("all tests passed")
