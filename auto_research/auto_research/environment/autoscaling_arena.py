# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Autoscaling Arena — the evaluation harness for the Planner.

The real Autoscaling Arena is not available in this environment, so this is a
self-contained simulator that plays the same role: it drives the current Planner
policy against a workload defined by the experiment config and returns an
``ArenaMetrics`` scorecard. Swap ``MockAutoscalingArena`` for a real
``ArenaAdapter`` implementation and the rest of the framework is unchanged.

Why the numbers respond to policy quality
-----------------------------------------
Replicas cannot be provisioned instantly: ``config.provision_step`` caps how
many can be added or removed per control step (decode replicas take time to
spin up). So a policy that reacts only after load has arrived spends several
steps overloaded — TTFT/ITL blow past the SLOs — while a policy that forecasts
load pre-provisions and rides the burst out. That tension, plus a GPU-cost term,
is what the score rewards.
"""

from __future__ import annotations

import math
import random
import traceback
from dataclasses import dataclass, field
from typing import List, Optional, Protocol

from ..schemas import ArenaMetrics, EvalMode, ExperimentConfig
from .planner import PlannerUnderTest


@dataclass
class PolicyContext:
    """Read-only view handed to the policy's ``decide`` each control step."""

    step: int
    current_replicas: int
    offered_rps: float
    recent_rps: List[float]
    utilization: float
    p_ttft_ms: float
    p_itl_ms: float
    per_replica_rps: float
    max_replicas: int
    ttft_slo_ms: float
    itl_slo_ms: float


@dataclass
class ArenaResult:
    """What the arena returns: either metrics, or an execution error to route."""

    metrics: Optional[ArenaMetrics] = None
    error: Optional[str] = None
    error_traceback: str = ""
    logs: str = ""
    trace_len: int = 0


class ArenaAdapter(Protocol):
    """The seam the real Autoscaling Arena would implement."""

    def run(
        self, planner: PlannerUnderTest, config: ExperimentConfig, mode: EvalMode
    ) -> ArenaResult:  # pragma: no cover - protocol
        ...


def build_workload(config: ExperimentConfig, mode: EvalMode) -> List[float]:
    """Deterministic offered-RPS trace: diurnal baseline + sharp bursts."""
    rng = random.Random(config.seed)
    steps = config.horizon_steps if mode == EvalMode.FULL else max(48, config.horizon_steps // 6)
    peak = config.per_replica_rps * config.max_replicas * 0.66  # keep headroom
    trough = peak * 0.12
    trace: List[float] = []
    burst_left = 0
    burst_gain = 1.0
    for t in range(steps):
        # diurnal sine over the horizon (one full day)
        phase = 2 * math.pi * t / max(1, steps)
        diurnal = trough + (peak - trough) * (0.5 - 0.5 * math.cos(phase))
        # small persistent noise
        noise = rng.uniform(-0.04, 0.04) * peak
        rps = max(1.0, diurnal + noise)
        # Sustained bursts: load stays elevated for several steps, so a policy
        # that reads the rising trend and provisions ahead of the (capped)
        # provisioning rate rides them out, while a purely reactive one lags.
        if burst_left == 0 and rng.random() < 0.06:
            burst_left = rng.randint(4, 8)
            burst_gain = rng.uniform(1.4, 1.8)
        if burst_left > 0:
            rps *= burst_gain
            burst_left -= 1
        trace.append(rps)
    return trace


def _latency_ms(offered: float, replicas: int, config: ExperimentConfig):
    """Queueing-style latency model: fine under capacity, steep once overloaded."""
    capacity = max(1e-9, replicas * config.per_replica_rps)
    lf = offered / capacity
    if lf <= 1.0:
        ttft = config.base_ttft_ms * (1.0 + 1.5 * lf * lf)
        itl = config.base_itl_ms * (1.0 + 0.8 * lf)
    else:
        over = lf - 1.0
        ttft = config.base_ttft_ms * 2.5 * (1.0 + 8.0 * over)
        itl = config.base_itl_ms * 1.8 * (1.0 + 5.0 * over)
    return ttft, itl, lf


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered)) - 1))
    return ordered[idx]


class MockAutoscalingArena:
    """Stand-in for the real Arena. Deterministic given the config seed."""

    def run(
        self, planner: PlannerUnderTest, config: ExperimentConfig, mode: EvalMode
    ) -> ArenaResult:
        try:
            decide = planner.load_decider()
        except Exception as exc:  # syntax/name/attr error in generated policy
            return ArenaResult(
                error=f"Planner failed to load: {type(exc).__name__}: {exc}",
                error_traceback=traceback.format_exc(),
            )

        trace = build_workload(config, mode)
        replicas = max(1, int(math.ceil(trace[0] / config.per_replica_rps)))
        recent: List[float] = []
        ttfts: List[float] = []
        itls: List[float] = []
        ok_steps = 0
        ttft_viol = 0
        itl_viol = 0
        scaling_actions = 0
        replica_steps = 0.0
        prev_ttft = config.base_ttft_ms
        prev_itl = config.base_itl_ms
        log_lines: List[str] = []
        capture_logs = mode == EvalMode.FOCUSED

        for t, offered in enumerate(trace):
            capacity = replicas * config.per_replica_rps
            utilization = offered / max(1e-9, capacity)
            ctx = PolicyContext(
                step=t,
                current_replicas=replicas,
                offered_rps=offered,
                recent_rps=list(recent[-8:]),
                utilization=utilization,
                p_ttft_ms=prev_ttft,
                p_itl_ms=prev_itl,
                per_replica_rps=config.per_replica_rps,
                max_replicas=config.max_replicas,
                ttft_slo_ms=config.ttft_slo_ms,
                itl_slo_ms=config.itl_slo_ms,
            )
            try:
                target = int(decide(ctx))
            except Exception as exc:  # policy raised at runtime
                return ArenaResult(
                    error=f"Planner raised at step {t}: {type(exc).__name__}: {exc}",
                    error_traceback=traceback.format_exc(),
                )

            target = max(1, min(config.max_replicas, target))
            # provisioning lag: move toward target, capped per step
            if target > replicas:
                new_replicas = min(target, replicas + config.provision_step)
            else:
                new_replicas = max(target, replicas - config.provision_step)
            if new_replicas != replicas:
                scaling_actions += 1
            replicas = new_replicas

            # serve this step with the (lagged) replica count
            ttft, itl, _lf = _latency_ms(offered, replicas, config)
            prev_ttft, prev_itl = ttft, itl
            ttfts.append(ttft)
            itls.append(itl)
            replica_steps += replicas
            recent.append(offered)

            ttft_ok = ttft <= config.ttft_slo_ms
            itl_ok = itl <= config.itl_slo_ms
            if not ttft_ok:
                ttft_viol += 1
            if not itl_ok:
                itl_viol += 1
            if ttft_ok and itl_ok:
                ok_steps += 1

            if capture_logs and (not ttft_ok or not itl_ok):
                log_lines.append(
                    f"step={t:>3} rps={offered:6.1f} replicas={replicas:>2} "
                    f"util={utilization:4.2f} ttft={ttft:6.1f}ms itl={itl:5.1f}ms "
                    f"{'TTFT!' if not ttft_ok else ''}{'ITL!' if not itl_ok else ''}"
                )

        n = len(trace)
        sla_attainment = ok_steps / n
        cost_cap = config.max_replicas * n
        norm_cost = replica_steps / max(1e-9, cost_cap)
        score = 0.75 * sla_attainment + 0.25 * (1.0 - norm_cost)

        metrics = ArenaMetrics(
            sla_attainment=round(sla_attainment, 4),
            p95_ttft_ms=round(_p95(ttfts), 1),
            p95_itl_ms=round(_p95(itls), 1),
            ttft_violations=ttft_viol,
            itl_violations=itl_viol,
            gpu_replica_steps=round(replica_steps, 1),
            scaling_actions=scaling_actions,
            mean_replicas=round(replica_steps / n, 2),
            score=round(score, 4),
        )
        logs = "\n".join(log_lines[:40])
        if capture_logs and len(log_lines) > 40:
            logs += f"\n... ({len(log_lines) - 40} more violating steps)"
        return ArenaResult(metrics=metrics, logs=logs, trace_len=n)
