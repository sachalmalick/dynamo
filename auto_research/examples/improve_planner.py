# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the Auto Research loop programmatically and inspect the artifacts.

    python examples/improve_planner.py

Uses the heuristic backend by default (no API key needed). Set AUTO_RESEARCH_MODE=llm
plus a provider key to run the real LangChain agents.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from auto_research import ExperimentConfig, ResearchLoop, Settings


def emit(actor: str, arrow: str, text: str) -> None:
    print(f"  {actor:<18} {arrow} {text}")


def main() -> None:
    settings = Settings.from_env()
    config = ExperimentConfig(
        name="planner-autoscaling-study",
        workload="diurnal_with_bursts",
        horizon_steps=288,
        max_iterations=4,
        target_score=0.87,
    )

    print(f"Running Auto Research (mode={settings.mode})\n")
    loop = ResearchLoop(config, settings, emit=emit)
    report = loop.run()

    print("\n=== Research Report ===")
    print(f"accepted policy : {report.accepted_change}")
    print(f"baseline score  : {report.baseline_score:.3f}")
    print(f"best score      : {report.best_score:.3f}")
    print("\n=== Final Planner policy ===")
    print(loop.planner.read_source())


if __name__ == "__main__":
    main()
