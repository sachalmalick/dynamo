# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LangChain chat-model factory.

Isolated here so the rest of the package imports cleanly even when LangChain is
not installed (the heuristic backend never calls this). ``make_llm`` returns a
``BaseChatModel`` that supports ``.with_structured_output`` and ``.bind_tools``,
which is all the agents rely on.
"""

from __future__ import annotations

from .config import Settings


def make_llm(settings: Settings):
    """Build a LangChain chat model for the configured provider.

    Raises a helpful error if the LangChain integration package is missing.
    """
    if settings.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "llm mode with provider=anthropic needs `langchain-anthropic`. "
                "Install it (`pip install -r auto_research/requirements.txt`) or set "
                "AUTO_RESEARCH_MODE=heuristic to run offline."
            ) from exc
        return ChatAnthropic(model=settings.model, temperature=settings.temperature)

    if settings.provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:  # pragma: no cover - depends on env
            raise ImportError(
                "llm mode with provider=openai needs `langchain-openai`. "
                "Install it or set AUTO_RESEARCH_MODE=heuristic to run offline."
            ) from exc
        return ChatOpenAI(model=settings.model, temperature=settings.temperature)

    raise ValueError(f"Unknown provider: {settings.provider!r}")
