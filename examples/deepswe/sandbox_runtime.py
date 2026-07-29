"""Sandbox runtime wrapper for SWE-bench / R2E-Gym in GKE environment.

Strictly imports and uses official `agent_sandbox_rl.adapters.r2egym` from
`kubernetes-sigs/agent-sandbox`.
"""

from absl import logging

try:
  from agent_sandbox_rl.adapters.r2egym import (
      make_fleet_repo_env,
      r2egym_command_files,
      _build_classes,
  )
  _docker_mod, _EnvArgs, _RepoEnv = _build_classes()
  FleetDockerRuntime = _docker_mod.FleetDockerRuntime
except Exception as e:
  raise ImportError(
      "use_agent_sandbox=True strictly requires the 'agent_sandbox_rl' package"
      " from kubernetes-sigs/agent-sandbox."
  ) from e


__all__ = [
    "FleetDockerRuntime",
    "make_fleet_repo_env",
    "r2egym_command_files",
]
