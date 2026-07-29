# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime settings for the framework.

Two execution backends are supported and share one set of agents, tools, and the
LangGraph wiring:

* ``llm``       - agents are LangChain models (Anthropic / OpenAI) that reason,
                  generate policy code, and author reports. Needs an API key.
* ``heuristic`` - deterministic stand-ins for each agent. No key required, so the
                  full loop runs offline and in CI. Same graph, same environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Settings:
    mode: str = "heuristic"  # "llm" | "heuristic"
    provider: str = "anthropic"  # "anthropic" | "openai"
    model: str = "claude-sonnet-5"
    temperature: float = 0.0
    workspace_dir: str = ""  # where the editable Planner policy lives
    max_tool_iterations: int = 6
    verbose: bool = True

    @classmethod
    def from_env(cls) -> "Settings":
        mode = os.getenv("AUTO_RESEARCH_MODE", "").strip().lower()
        if mode not in ("llm", "heuristic"):
            # Auto-detect: use the LLM backend only if a key is actually present.
            has_key = bool(
                os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
            )
            mode = "llm" if has_key else "heuristic"

        provider = os.getenv("AUTO_RESEARCH_PROVIDER", "anthropic").strip().lower()
        default_model = (
            "claude-sonnet-5" if provider == "anthropic" else "gpt-4.1"
        )
        return cls(
            mode=mode,
            provider=provider,
            model=os.getenv("AUTO_RESEARCH_MODEL", default_model),
            temperature=float(os.getenv("AUTO_RESEARCH_TEMPERATURE", "0.0")),
            workspace_dir=os.getenv("AUTO_RESEARCH_WORKSPACE", ""),
            max_tool_iterations=int(os.getenv("AUTO_RESEARCH_MAX_TOOL_ITERS", "6")),
            verbose=os.getenv("AUTO_RESEARCH_VERBOSE", "1") not in ("0", "false", ""),
        )
