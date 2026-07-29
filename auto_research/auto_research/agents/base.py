# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the three agents."""

from __future__ import annotations

from typing import List, Optional

from ..schemas import ArenaMetrics, ExperimentReport


def verdict_for(delta: float) -> str:
    """Classify a score change relative to the baseline."""
    if delta > 0.01:
        return "improved"
    if delta < -0.01:
        return "regressed"
    return "neutral"


def summarize_metrics(m: ArenaMetrics) -> str:
    return (
        f"score={m.score:.3f} sla={m.sla_attainment:.1%} "
        f"p95_ttft={m.p95_ttft_ms:.0f}ms p95_itl={m.p95_itl_ms:.1f}ms "
        f"viol(ttft/itl)={m.ttft_violations}/{m.itl_violations} "
        f"gpu_steps={m.gpu_replica_steps:.0f} mean_replicas={m.mean_replicas:.1f}"
    )


def run_react_agent(model, tools, system: str, user: str, max_iterations: int) -> str:
    """Run a LangChain/LangGraph ReAct tool-loop and return the final text.

    Kept in one place so both tool-using agents share the same wiring.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.prebuilt import create_react_agent

    agent = create_react_agent(model, tools)
    result = agent.invoke(
        {"messages": [SystemMessage(content=system), HumanMessage(content=user)]},
        config={"recursion_limit": 2 * max_iterations + 3},
    )
    messages = result["messages"]
    for msg in reversed(messages):
        content = getattr(msg, "content", "")
        if isinstance(content, list):  # some providers return content blocks
            content = " ".join(
                block.get("text", "") for block in content if isinstance(block, dict)
            )
        if content and getattr(msg, "type", "") in ("ai", "assistant"):
            return content
    return ""


def history_digest(history: List[dict], limit: int = 8) -> str:
    """Compact, human-readable recap of what has happened so far."""
    lines = []
    for ev in history[-limit:]:
        lines.append(f"- {ev.get('actor', '?')}: {ev.get('summary', '')}")
    return "\n".join(lines) if lines else "(nothing yet)"
