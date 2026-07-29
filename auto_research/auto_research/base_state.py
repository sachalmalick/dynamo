# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared loop-state type and the terminal sentinel.

Isolated from ``graph.py`` so importing the state schema never pulls in
LangGraph. ``LoopState`` is a plain ``TypedDict`` (LangGraph accepts it as a
graph schema; the fallback driver treats it as an ordinary dict).
"""

from __future__ import annotations

from typing import Any, List, Optional, TypedDict

from .schemas import (
    ChangeRequest,
    CodeChange,
    EvalRequest,
    ExecutionError,
    ExperimentConfig,
    ExperimentReport,
    LeadDecision,
    ResearchReport,
)

# Terminal marker for the fallback driver; mapped to langgraph.graph.END when the
# real graph is compiled.
END = "__end__"


class LoopState(TypedDict, total=False):
    config: ExperimentConfig
    iteration: int
    backlog_index: int
    baseline: Optional[ExperimentReport]
    best: Optional[ExperimentReport]
    best_strategy: str
    latest_report: Optional[ExperimentReport]
    pending_change: Optional[ChangeRequest]
    pending_eval: Optional[EvalRequest]
    pending_error: Optional[ExecutionError]
    last_change: Optional[CodeChange]
    last_strategy: str
    fix_attempts: int
    decision: Optional[LeadDecision]
    history: List[Any]
    final: Optional[ResearchReport]
