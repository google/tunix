"""Strict, CPU-only recipe resolution. No model imports or accelerator access.

The checked-in E2B YAML is the workload authority. Overrides here are limited
to the arm, four-device shared topology, artifact paths, and a stage stop.
The stock example and all existing Qwen recipes remain unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO = Path(__file__).resolve().parents[3]
MODEL_ID = "google/gemma-4-E2B-it"
IMAGE_ID = "sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a"
VERSIONS = {"vllm-tpu": "0.25.0", "tpu_inference": "0.25.0",
            "flax": "0.12.4", "jax": "0.10.2", "jaxlib": "0.10.2"}
SOURCE_FILES = {
    "examples/frozenlake/configs/gemma4_e2b.yaml":
        "c8b5934e00889bd86dafb5a68958f3b08864fc65d369933300f31096fc710456",
    "tunix/cli/base_agentic_config.yaml":
        "acaa3ef2acf8971e232ade5ce414906f87ffb01c7c3ba52eb36fff0f124019bb",
}
ARMS = ("native", "tis", "zero")
STAGES = ("stock-admission", "train")


class RecipeError(ValueError):
  """Invalid or contaminated comparison identity."""


def json_bytes(value: Any) -> bytes:
  return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False) + "\n").encode()


def digest(value: Any) -> str:
  return hashlib.sha256(json_bytes(value)).hexdigest()


def require_isolated_environment(env: Mapping[str, str]) -> None:
  # Presence matters: even an empty/zero foreign selector is rejected. Never
  # include values in errors because the caller's environment can hold secrets.
  foreign = sorted(k for k in env if k.startswith(("CANON_", "FL_", "T_")))
  foreign += sorted(k for k in ("SHARED_MESH_SHAPE", "ROLLOUT_ENGINE") if k in env)
  if foreign:
    raise RecipeError("foreign recipe environment keys: " + ",".join(foreign))
  if env.get("JAX_PLATFORMS", "") not in ("", "cpu", "tpu"):
    raise RecipeError("only direct one-host or CPU construction is supported")


def require_versions() -> dict[str, str]:
  actual = {name: importlib.metadata.version(name) for name in VERSIONS}
  if actual != VERSIONS:
    raise RecipeError("installed runtime versions differ from pinned contract")
  return actual


def _yaml(relative: str, repo: Path) -> dict[str, Any]:
  raw = (repo / relative).read_bytes()
  if hashlib.sha256(raw).hexdigest() != SOURCE_FILES[relative]:
    raise RecipeError("default recipe source drift: " + relative)
  return yaml.safe_load(raw)


def _merge(base: dict, override: dict) -> dict:
  result = copy.deepcopy(base)
  for key, value in override.items():
    if isinstance(value, dict) and isinstance(result.get(key), dict):
      result[key] = _merge(result[key], value)
    else:
      result[key] = copy.deepcopy(value)
  return result


def resolve(arm: str, *, stage: str = "train", repo: Path = REPO) -> dict:
  if arm not in ARMS or stage not in STAGES:
    raise RecipeError("unknown Gemma arm or stage")
  if stage == "stock-admission" and arm == "zero":
    raise RecipeError("stock admission is not a Zero-TIM stage")
  base = _yaml("tunix/cli/base_agentic_config.yaml", repo)
  default = _yaml("examples/frozenlake/configs/gemma4_e2b.yaml", repo)
  # PyYAML calls the YAML spelling 1e-6 a string; the real CLI's OmegaConf
  # schema consumes it as a float. Preserve the consumer value explicitly.
  default["rl_training_config"]["actor_optimizer_config"]["peak_value"] = float(
      default["rl_training_config"]["actor_optimizer_config"]["peak_value"])
  config = _merge(base, default)
  # HyperParameters replaces this whole section; recursively merging it would
  # leak gradient_accumulation_steps=1 and fail RLTrainingConfig construction.
  config["rl_training_config"] = copy.deepcopy(default["rl_training_config"])
  # Match HyperParameters' base-to-role inheritance without preserving a
  # stale Llama model or accidentally allocating three separate TP4 meshes.
  for role in ("actor", "reference", "rollout"):
    role_key = role + "_model_config"
    config[role_key] = _merge(config["model_config"], default[role_key])
  for key in ("model_config", "actor_model_config"):
    config[key]["mesh"] = {
        "shape": "(1,4)", "axis_names": "('fsdp','tp')",
        "allocation_policy": "COMPACT",
    }
  for key in ("reference_model_config", "rollout_model_config"):
    config[key]["mesh"] = None
    config[key]["same_mesh_as"] = "actor"
  config["vllm_config"]["data_parallel_size"] = 1
  config["vllm_config"]["tensor_parallel_size"] = 4
  config["agentic_grpo_config"]["sampler_is"] = "token" if arm == "tis" else None
  config["agentic_grpo_config"]["use_rollout_logps"] = True
  # The inherited pipeline resolves explicit warmup_steps=0 through its
  # existing falsey rule. Preserve that behavior; record the resolved value
  # after compute_params instead of silently fixing the schedule here.
  contract = {
      "schema": "gemma4-e2b-default-tim-v1", "model_id": MODEL_ID,
      "arm": arm, "stage": stage, "expected_updates": 0 if stage == "stock-admission" else 5,
      "old_logps": "trainer" if arm == "tis" else "rollout",
      "correction": "detached-token-tis-2" if arm == "tis" else "none",
      "topology": {"hosts": 1, "devices": 4, "dp": 1, "tp": 4},
      "image_config_id": IMAGE_ID, "versions": VERSIONS.copy(),
      "source_files": SOURCE_FILES.copy(), "config": config,
      "target_status": "NOT_RUN",
  }
  contract["contract_sha256"] = digest(contract)
  return contract


def validate_resolved(contract: dict) -> None:
  expected = resolve(contract.get("arm"), stage=contract.get("stage"))
  if contract != expected:
    raise RecipeError("resolved contract differs from registered default recipe")


def bind_paths(config: dict, *, snapshot: Path, data: Path, output: Path) -> dict:
  """Bind artifact locations without changing model, loss or workload fields."""
  result = copy.deepcopy(config)
  result["tokenizer_config"]["tokenizer_path"] = str(snapshot)
  result["vllm_config"]["model_version"] = str(snapshot)
  result["data_config"]["data_dir"] = str(data)
  train = result["rl_training_config"]
  train["metrics_logging_options"]["log_dir"] = str(output / "metrics")
  train["checkpoint_root_directory"] = str(output / "checkpoints")
  return result
