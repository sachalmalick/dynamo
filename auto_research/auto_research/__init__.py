# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto Research — a LangChain multi-agent framework that autonomously improves
the Dynamo Planner's autoscaling policy by looping SWE edits through an
Autoscaling Arena, coordinated by a Research Lead.

Public entry points are kept import-light: ``ResearchLoop`` and the schemas pull
in no heavy optional dependency at import time.
"""

from .config import Settings
from .graph import ResearchLoop
from .schemas import (
    ChangeRequest,
    EvalRequest,
    ExecutionError,
    ExperimentConfig,
    ExperimentReport,
    LeadDecision,
    ResearchReport,
)

__all__ = [
    "ResearchLoop",
    "Settings",
    "ExperimentConfig",
    "ChangeRequest",
    "EvalRequest",
    "ExecutionError",
    "ExperimentReport",
    "LeadDecision",
    "ResearchReport",
]

__version__ = "0.1.0"
