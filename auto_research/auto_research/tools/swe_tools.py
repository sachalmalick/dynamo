# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWE Agent tools: inspect and edit the Planner policy in the workspace."""

from __future__ import annotations

from typing import Dict, List

from langchain_core.tools import tool

from ..environment import PlannerUnderTest


def make_swe_tools(planner: PlannerUnderTest) -> List:
    """Build read/write tools bound to a specific Planner workspace."""

    @tool
    def read_planner_source() -> str:
        """Return the full current source of the Planner autoscaling policy.

        The policy must define ``decide(ctx) -> int`` returning the target
        replica count. Read this before editing so you preserve the contract.
        """
        return planner.read_source()

    @tool
    def write_planner_source(source: str) -> str:
        """Overwrite the Planner policy with new Python source.

        ``source`` must be a complete module defining ``decide(ctx) -> int``.
        Returns a short status string. Syntax errors are only detected when the
        Autoscaling Arena imports the module during evaluation.
        """
        planner.write_source(source)
        return f"wrote {len(source)} chars to {planner.policy_path}"

    return [read_planner_source, write_planner_source]
