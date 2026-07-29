# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry point: ``python -m auto_research``.

Runs the full multi-agent loop and prints the message flow between agents (the
arrows in the reference diagram) followed by the final PR + Research Report.
"""

from __future__ import annotations

import argparse
import sys

from .config import Settings
from .graph import ResearchLoop
from .schemas import ExperimentConfig


def _emitter(verbose: bool):
    def emit(actor: str, arrow: str, text: str) -> None:
        if verbose:
            print(f"  {actor:<18} {arrow} {text}")

    return emit


def build_config(args) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if args.workload:
        cfg.workload = args.workload
    if args.horizon:
        cfg.horizon_steps = args.horizon
    if args.max_iterations:
        cfg.max_iterations = args.max_iterations
    if args.target_score is not None:
        cfg.target_score = args.target_score
    return cfg


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="auto_research",
        description="Autonomous multi-agent loop that improves the Dynamo Planner "
        "autoscaling policy via an Autoscaling Arena.",
    )
    parser.add_argument("--mode", choices=["llm", "heuristic"], default=None,
                        help="agent backend (default: auto-detect from API keys)")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--workspace", default=None, help="Planner workspace dir")
    parser.add_argument("--workload", default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--target-score", type=float, default=None)
    parser.add_argument("--no-langgraph", action="store_true",
                        help="use the dependency-free driver even if langgraph is present")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    if args.mode:
        settings.mode = args.mode
    if args.provider:
        settings.provider = args.provider
    if args.model:
        settings.model = args.model
    if args.workspace:
        settings.workspace_dir = args.workspace
    if args.quiet:
        settings.verbose = False

    config = build_config(args)

    print("=" * 78)
    print(f"Auto Research  |  mode={settings.mode}  workload={config.workload}  "
          f"max_iters={config.max_iterations}")
    print("=" * 78)
    if not args.quiet:
        print("\nMessage flow:")

    loop = ResearchLoop(config, settings, emit=_emitter(not args.quiet))
    report = loop.run(use_langgraph=not args.no_langgraph)

    print("\n" + "=" * 78)
    print("RESEARCH REPORT")
    print("=" * 78)
    print(f"Title    : {report.title}")
    print(f"Baseline : {report.baseline_score:.3f}")
    print(f"Best     : {report.best_score:.3f}  "
          f"(Δ{report.best_score - report.baseline_score:+.3f})")
    print(f"Accepted : {report.accepted_change}")
    print(f"Iters    : {report.iterations}")
    print(f"\nSummary  : {report.summary}")
    if report.findings:
        print("\nPer-iteration scores:")
        for f in report.findings:
            print(f"  - {f}")
    print("\n" + "-" * 78)
    print("PROPOSED PULL REQUEST")
    print("-" * 78)
    print(f"{report.pr_title}\n")
    print(report.pr_body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
