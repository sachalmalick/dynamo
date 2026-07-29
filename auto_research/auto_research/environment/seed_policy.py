# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed autoscaling policy — the Planner artifact the SWE Agent edits.

This is the code that gets copied into the Execution Environment workspace and
mutated across iterations. It is deliberately small and self-contained so a
single edit is a meaningful research step.

Contract
--------
Implement ``decide(ctx) -> int`` returning the *target* number of decode
replicas for the next control step. The Autoscaling Arena clamps how fast the
target can actually be provisioned, so a policy that reacts only after load has
already arrived will lag bursts and violate SLAs.

``ctx`` attributes (see arena.PolicyContext):
    step            int    control-step index
    current_replicas int   replicas serving right now
    offered_rps     float  load observed this step
    recent_rps      list   recent offered_rps history (most recent last)
    utilization     float  offered_rps / (current_replicas * per_replica_rps)
    p_ttft_ms       float  last observed p95 TTFT
    p_itl_ms        float  last observed p95 ITL
    per_replica_rps float
    max_replicas    int
    ttft_slo_ms     float
    itl_slo_ms      float
"""


def decide(ctx) -> int:
    # Reactive threshold on utilization: add a replica when hot, drop one when cold.
    target = ctx.current_replicas
    if ctx.utilization > 0.9:
        target += 1
    elif ctx.utilization < 0.5:
        target -= 1
    return max(1, min(ctx.max_replicas, target))
