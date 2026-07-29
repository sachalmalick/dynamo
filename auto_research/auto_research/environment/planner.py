# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Planner artifact inside the Execution Environment.

``PlannerUnderTest`` owns the editable policy file. The SWE Agent writes to it
via tools; the Autoscaling Arena imports it fresh on every eval so code changes
take effect. This is the sandboxed working copy — it never touches the real
``components/src/dynamo/planner`` tree.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from types import ModuleType

_SEED = os.path.join(os.path.dirname(__file__), "seed_policy.py")

POLICY_FILENAME = "planner_policy.py"


class PlannerUnderTest:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = os.path.abspath(workspace_dir)
        os.makedirs(self.workspace_dir, exist_ok=True)
        self._load_counter = 0

    @property
    def policy_path(self) -> str:
        return os.path.join(self.workspace_dir, POLICY_FILENAME)

    def reset_to_seed(self) -> None:
        """Copy the seed policy into the workspace, discarding prior edits."""
        shutil.copyfile(_SEED, self.policy_path)

    def ensure_seeded(self) -> None:
        if not os.path.exists(self.policy_path):
            self.reset_to_seed()

    def read_source(self) -> str:
        self.ensure_seeded()
        with open(self.policy_path, "r", encoding="utf-8") as fh:
            return fh.read()

    def write_source(self, source: str) -> None:
        with open(self.policy_path, "w", encoding="utf-8") as fh:
            fh.write(source)

    def load_decider(self):
        """Import ``decide`` from the current policy file, bypassing any cache.

        A fresh module name per load guarantees edits are picked up and that a
        syntax error surfaces here (so the Arena can route it as an
        ExecutionError) rather than serving a stale build.
        """
        self.ensure_seeded()
        self._load_counter += 1
        mod_name = f"_planner_policy_{self._load_counter}"
        spec = importlib.util.spec_from_file_location(mod_name, self.policy_path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise ImportError(f"cannot load policy from {self.policy_path}")
        module: ModuleType = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # may raise SyntaxError / NameError
        if not hasattr(module, "decide"):
            raise AttributeError(
                "policy module does not define `decide(ctx) -> int`"
            )
        return module.decide
