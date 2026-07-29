# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The three agents from the reference diagram."""

from .eval_agent import EvalAgent
from .research_lead import ResearchLeadAgent
from .swe_agent import SWEAgent

__all__ = ["ResearchLeadAgent", "SWEAgent", "EvalAgent"]
