# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LangChain tools the SWE and Eval agents act on the Execution Environment with.

The tools are thin, authoritative wrappers over the environment: reading and
writing the Planner source, and running the Autoscaling Arena. The LLM agents
call them during reasoning; the deterministic results (file contents, arena
metrics) are what the graph nodes trust — never the model's paraphrase of them.
"""

from .swe_tools import make_swe_tools
from .eval_tools import make_eval_tools

__all__ = ["make_swe_tools", "make_eval_tools"]
