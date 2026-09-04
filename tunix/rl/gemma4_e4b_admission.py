"""Fail-closed one-host admission contract for Gemma 4 E4B-IT.

This module contains no Zero-TIM implementation.  It admits the exact stock
model selected by the Gemma FrozenLake P0 identity freeze and records enough
runtime evidence to decide whether one v5p host can carry P1.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


SELECTOR_ENV = "CANON_GEMMA4_E4B_P1_ADMISSION"
MODEL_ID = "google/gemma-4-E4B-it"
MODEL_CONFIG_ID = "gemma4_e4b_it"
CHECKPOINT_REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"
CHECKPOINT_MANIFEST_SHA256 = (
    "ca1deed032455e0c7bf0fd4cd73a0c47a4841921f9dc99b2e98d00cb6c5f70b1"
)
RUNTIME_IMAGE_DIGEST = (
    "sha256:418dc632edd8ff990e8880df6a5ca82369f6c4d705e16152c1ee6f9708d5e53a"
)
EXPECTED_JAX_VERSION = "0.10.2"
TOKENIZER_CONFIG_SHA256 = (
    "9f4fec4b1dc6ecddf8f4a92e9caea5971c0e67d81309f3f9066a2bee8c362633"
)
CHAT_TEMPLATE_SHA256 = (
    "0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5"
)
TOKEN_PROBE_SHA256 = (
    "ed5d74c73d9f7c733738d1cf93fd41a8e1aa9720a49461485a34dc89a4c77f4a"
)
_TOKEN_PROBE_TEXTS = (
    "<bos>",
    "<|turn>user\n",
    "<|turn>model\n<|channel>thought\n<channel|>",
    "<turn|>\n",
    "LEFT RIGHT UP DOWN",
)
EXPECTED_LAYER_TYPES = tuple(
    "full_attention" if (index + 1) % 6 == 0 else "sliding_attention"
    for index in range(42)
)
SNAPSHOT_FILES = {
    "model.safetensors": {
        "bytes": 15_992_595_884,
        "sha256": "cfbd3d2f1cd71bd471c37fe2bf8546d5028d41e5736f64e1ca6c6b8893125503",
    },
    "config.json": {
        "bytes": 5_145,
        "sha256": "33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4",
    },
    "generation_config.json": {
        "bytes": 208,
        "sha256": "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
    },
    "tokenizer.json": {
        "sha256": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    },
    "tokenizer_config.json": {
        "sha256": TOKENIZER_CONFIG_SHA256,
    },
}

WORKLOADS: dict[str, dict[str, Any]] = {
    "p45": {
        "dataset_recipe": "legacy",
        "dataset_split": "legacy",
        "train_count": 10_000,
        "eval_count": 100,
        "train_sha256": (
            "ddc96fd9ae4e807d8aa8e800795aa743e423ffe4f936f681596460d28e670487"
        ),
        "eval_sha256": (
            "b10add7f31b2cc9931c65b4cc59780004fd3d52a4fce9d20ed565c87df44b580"
        ),
        "max_turns": 5,
        "max_prompt_length": 4096,
        "max_response_length": 2048,
        "context_hard_cap": 6400,
    },
    "m15": {
        "dataset_recipe": "m15",
        "dataset_split": "main",
        "train_count": 10_000,
        "eval_count": 100,
        "train_sha256": (
            "ff1e659b80a0c9bd640e616972a523132f4a333ef174b1a0b13b202958a30e43"
        ),
        "eval_sha256": (
            "8edb61cb995b4abe8d3f90b32e961be74b8b74ab46120e0d43513ea26d324089"
        ),
        "max_turns": 15,
        "max_prompt_length": 4096,
        "max_response_length": 8192,
        "context_hard_cap": 12_288,
    },
}

_FORBIDDEN_CLI_PREFIXES = (
    "--batch_size",
    "--mini_batch_size",
    "--num_batches",
    "--num_generations",
    "--max_prompt_length",
    "--max_response_length",
    "--max_concurrency",
)
_FORBIDDEN_ACTIVE_FLAGS = (
    "CANON_ALIGNMENT_GATE",
    "CANON_ENGINE_MODULE_C",
    "CANON_FIXED_AR",
    "CANON_FIXED_AR_EMBED",
    "CANON_FIXED_AR_GATHER",
    "CANON_P38_FIXED_LM_HEAD",
    "CANON_P59_CHECKED_VMA",
    "CANON_P59_RANK_PARALLEL_BACKWARD",
    "CANON_PALLAS_ALL_PROJ",
    "CANON_PALLAS_ALL_RMSNORM",
    "CANON_PALLAS_CANONICAL_VJP",
    "CANON_PALLAS_LOGSOFTMAX",
    "CANON_RPA_VJP2",
)


def active_workload(environ: Mapping[str, str] | None = None) -> str | None:
  """Returns the admitted workload, with missing/empty/0 meaning disabled."""
  environ = os.environ if environ is None else environ
  raw = environ.get(SELECTOR_ENV)
  if raw in (None, "", "0"):
    return None
  if raw not in WORKLOADS:
    raise ValueError(
        f"{SELECTOR_ENV} must be absent, empty, 0, p45, or m15; got {raw!r}"
    )
  return raw


def workload_contract(name: str) -> dict[str, Any]:
  try:
    return dict(WORKLOADS[name])
  except KeyError as exc:
    raise ValueError(f"unknown Gemma E4B P1 workload {name!r}") from exc


def require_stock_invocation(
    argv: tuple[str, ...], environ: Mapping[str, str] | None = None
) -> None:
  """Rejects attempts to mix P1 with training knobs or canonical kernels."""
  environ = os.environ if environ is None else environ
  forbidden_args = [
      value
      for value in argv
      if any(
          value == prefix or value.startswith(prefix + "=")
          for prefix in _FORBIDDEN_CLI_PREFIXES
      )
  ]
  if forbidden_args:
    raise ValueError(f"Gemma E4B P1 forbids CLI recipe overrides: {forbidden_args}")
  active = [
      name
      for name in _FORBIDDEN_ACTIVE_FLAGS
      if environ.get(name) not in (None, "", "0")
  ]
  if active:
    raise ValueError(f"Gemma E4B P1 forbids canonical runtime flags: {active}")
  prefix_cache = environ.get("CANON_VLLM_ENABLE_PREFIX_CACHING")
  if prefix_cache not in (None, "", "0"):
    raise ValueError("Gemma E4B P1 requires prefix caching off")


def identity_receipt(workload: str) -> dict[str, Any]:
  contract = workload_contract(workload)
  return {
      "schema": "gemma4-e4b-p1-identity-v1",
      "workload": workload,
      "model_config_id": MODEL_CONFIG_ID,
      "checkpoint_revision": CHECKPOINT_REVISION,
      "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
      "tokenizer_config_sha256": TOKENIZER_CONFIG_SHA256,
      "dataset_train_sha256": contract["train_sha256"],
      "dataset_eval_sha256": contract["eval_sha256"],
      "mesh": {"dp": 1, "tp": 4},
      "prefix_caching": False,
      "backward": 0,
      "optimizer_commits": 0,
  }


def verify_snapshot_identity(snapshot_dir: str) -> dict[str, Any]:
  """Hashes the exact P0-selected payload before either loader consumes it."""
  root = Path(snapshot_dir)
  files = {}
  for relative, expected in SNAPSHOT_FILES.items():
    path = root / relative
    if not path.is_file():
      raise RuntimeError(f"Gemma E4B P1 snapshot is missing {relative}")
    size = path.stat().st_size
    if "bytes" in expected and size != expected["bytes"]:
      raise RuntimeError(
          f"Gemma E4B P1 {relative} size drifted: {size} != {expected['bytes']}"
      )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
      for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected["sha256"]:
      raise RuntimeError(
          f"Gemma E4B P1 {relative} SHA-256 drifted: {actual}"
      )
    files[relative] = {"bytes": size, "sha256": actual}
  return {
      "schema": "gemma4-e4b-p1-snapshot-v1",
      "checkpoint_revision": CHECKPOINT_REVISION,
      "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
      "files": files,
  }


def verify_tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
  """Verifies the loaded tokenizer semantics frozen by the P0 probe."""
  token_probe = {
      value: [int(token) for token in tokenizer.encode(
          value, add_special_tokens=False
      )]
      for value in _TOKEN_PROBE_TEXTS
  }
  probe_sha256 = hashlib.sha256(
      json.dumps(
          token_probe, sort_keys=True, separators=(",", ":")
      ).encode("utf-8")
  ).hexdigest()
  chat_template_sha256 = hashlib.sha256(
      (tokenizer.chat_template or "").encode("utf-8")
  ).hexdigest()
  receipt = {
      "schema": "gemma4-e4b-p1-tokenizer-v1",
      "class": type(tokenizer).__name__,
      "vocab_size": len(tokenizer),
      "bos_id": tokenizer.bos_token_id,
      "eos_id": tokenizer.eos_token_id,
      "pad_id": tokenizer.pad_token_id,
      "unk_id": tokenizer.unk_token_id,
      "chat_template_sha256": chat_template_sha256,
      "token_probe_sha256": probe_sha256,
  }
  expected = {
      "class": "GemmaTokenizer",
      "vocab_size": 262144,
      "bos_id": 2,
      "eos_id": 1,
      "pad_id": 0,
      "unk_id": 3,
      "chat_template_sha256": CHAT_TEMPLATE_SHA256,
      "token_probe_sha256": TOKEN_PROBE_SHA256,
  }
  drift = {
      key: {"actual": receipt[key], "expected": value}
      for key, value in expected.items()
      if receipt[key] != value
  }
  if drift:
    raise RuntimeError(f"Gemma E4B P1 tokenizer identity drifted: {drift}")
  return receipt


def require_runtime_contract(
    *,
    workload: str,
    device_count: int,
    local_device_count: int,
    process_count: int,
    platforms: tuple[str, ...],
    device_kinds: tuple[str, ...],
    device_ids: tuple[int, ...],
    device_process_indices: tuple[int, ...],
    rollout_mesh_shape: tuple[int, int],
    trainer_mesh_shape: tuple[int, int],
    jax_version: str,
    checkpoint_root: str | None,
    prefix_caching: bool,
) -> dict[str, Any]:
  """Validates the target-only execution envelope and returns its receipt."""
  workload_contract(workload)
  errors = []
  if device_count != 4 or local_device_count != 4:
    errors.append(
        f"device_count/local_device_count must be 4/4, got "
        f"{device_count}/{local_device_count}"
    )
  if process_count != 1:
    errors.append(f"process_count must be 1, got {process_count}")
  if platforms != ("tpu",):
    errors.append(f"device platforms must be only tpu, got {platforms}")
  if device_kinds != ("TPU v5",):
    errors.append(f"device kind must be TPU v5, got {device_kinds}")
  if len(device_ids) != 4 or len(set(device_ids)) != 4:
    errors.append(f"device IDs must contain four unique values, got {device_ids}")
  if device_process_indices != (0, 0, 0, 0):
    errors.append(
        "all four devices must belong to process 0, got "
        f"{device_process_indices}"
    )
  if rollout_mesh_shape != (1, 4) or trainer_mesh_shape != (1, 4):
    errors.append(
        "rollout/trainer meshes must both be DP1xTP4, got "
        f"{rollout_mesh_shape}/{trainer_mesh_shape}"
    )
  if jax_version != EXPECTED_JAX_VERSION:
    errors.append(
        f"JAX version must be {EXPECTED_JAX_VERSION}, got {jax_version}"
    )
  if checkpoint_root is not None:
    errors.append("P1 admission forbids checkpoint writes")
  if prefix_caching:
    errors.append("P1 admission requires prefix caching off")
  if errors:
    raise RuntimeError("Gemma E4B P1 runtime contract failed: " + "; ".join(errors))
  return {
      **identity_receipt(workload),
      "device_count": device_count,
      "local_device_count": local_device_count,
      "process_count": process_count,
      "platforms": platforms,
      "device_kinds": device_kinds,
      "device_ids": device_ids,
      "device_process_indices": device_process_indices,
      "rollout_mesh_shape": rollout_mesh_shape,
      "trainer_mesh_shape": trainer_mesh_shape,
      "jax_version": jax_version,
      "target_geometry": "DP1xTP4",
  }


def shape_ledger(
    workload: str,
    *,
    semantic_prompts: int,
    generations_per_prompt: int,
    mini_batch_size: int,
    train_micro_batch_size: int,
    compute_logps_micro_batch_size: int,
    max_num_seqs: int,
    max_num_batched_tokens: int,
    kv_cache_size: int,
) -> dict[str, Any]:
  contract = workload_contract(workload)
  global_rows = semantic_prompts * generations_per_prompt
  if semantic_prompts != 1 or generations_per_prompt != 1:
    raise RuntimeError("P1 bounded admission requires exactly one rollout row")
  if (
      mini_batch_size,
      train_micro_batch_size,
      compute_logps_micro_batch_size,
  ) != (1, 1, 1):
    raise RuntimeError(
        "P1 trainer batch geometry must be mini/train/logps=1/1/1, got "
        f"{mini_batch_size}/{train_micro_batch_size}/"
        f"{compute_logps_micro_batch_size}"
    )
  if max_num_seqs < global_rows:
    raise RuntimeError("vLLM max_num_seqs is smaller than the semantic row count")
  if kv_cache_size < contract["context_hard_cap"]:
    raise RuntimeError(
        f"KV capacity {kv_cache_size} is below {workload} hard cap "
        f"{contract['context_hard_cap']}"
    )
  return {
      "schema": "gemma4-e4b-p1-shape-ledger-v1",
      "workload": workload,
      "semantic_prompts": semantic_prompts,
      "generations_per_prompt": generations_per_prompt,
      "caller_global_rows": global_rows,
      "dp": 1,
      "tp": 4,
      "shard_local_rows": global_rows,
      "mini_batch_size": mini_batch_size,
      "train_micro_batch_size": train_micro_batch_size,
      "compute_logps_micro_batch_size": compute_logps_micro_batch_size,
      "max_prompt_length": contract["max_prompt_length"],
      "max_response_length": contract["max_response_length"],
      "context_hard_cap": contract["context_hard_cap"],
      "vllm_max_num_seqs": max_num_seqs,
      "vllm_max_num_batched_tokens": max_num_batched_tokens,
      "kv_cache_size": kv_cache_size,
  }


def hbm_receipt(stage: str, devices: Any | None = None) -> dict[str, Any]:
  """Captures structured per-device HBM without fabricating unavailable data."""
  import jax  # Imported lazily so host contract tests do not initialize JAX.

  devices = tuple(jax.local_devices() if devices is None else devices)
  rows = []
  for device in devices:
    stats = device.memory_stats() or {}
    if "bytes_in_use" not in stats or "bytes_limit" not in stats:
      raise RuntimeError(f"HBM stats missing for device {device}")
    used = int(stats["bytes_in_use"])
    limit = int(stats["bytes_limit"])
    if used < 0 or limit <= 0 or used > limit:
      raise RuntimeError(
          f"invalid HBM stats for device {device}: used={used} limit={limit}"
      )
    rows.append({
        "device_id": int(device.id),
        "platform": str(device.platform),
        "bytes_in_use": used,
        "bytes_limit": limit,
        "bytes_free": limit - used,
        "fraction_in_use": used / limit,
    })
  receipt = {
      "schema": "gemma4-e4b-p1-hbm-v1",
      "stage": stage,
      "devices": rows,
      "min_bytes_free": min(row["bytes_free"] for row in rows),
  }
  print("[GEMMA4_E4B_P1_HBM] " + json.dumps(receipt, sort_keys=True), flush=True)
  return receipt


def optimizer_state_receipt(optimizer_state: Any) -> dict[str, Any]:
  """Proves that a nonempty optimizer state was physically materialized."""
  import jax
  import numpy as np

  jax.block_until_ready(optimizer_state)
  leaves = tuple(jax.tree_util.tree_leaves(optimizer_state))
  if not leaves:
    raise RuntimeError("Gemma E4B P1 optimizer state is empty")
  elements = 0
  allocated_bytes = 0
  memory_kinds: dict[str, int] = {}
  for leaf in leaves:
    elements += int(leaf.size)
    allocated_bytes += int(leaf.size) * int(np.dtype(leaf.dtype).itemsize)
    kind = str(getattr(getattr(leaf, "sharding", None), "memory_kind", None))
    memory_kinds[kind] = memory_kinds.get(kind, 0) + 1
  receipt = {
      "schema": "gemma4-e4b-p1-optimizer-state-v1",
      "leaves": len(leaves),
      "elements": elements,
      "allocated_bytes": allocated_bytes,
      "memory_kinds": memory_kinds,
  }
  print(
      "[GEMMA4_E4B_P1_OPTIMIZER] " + json.dumps(receipt, sort_keys=True),
      flush=True,
  )
  return receipt


def attest_exact_live_engine_weights(
    *, sampler: Any, trainer_state: Any, attest_fn: Any | None = None
) -> dict[str, Any]:
  """Runs the P1 Gemma-specific live-engine tensor identity gate."""
  if active_workload() is None:
    raise RuntimeError("Gemma E4B P1 attestation is not enabled")
  args = dict(getattr(sampler, "args", {}))
  if int(args.get("tensor_parallel_size", 0)) != 4:
    raise RuntimeError("Gemma E4B P1 live engine must use TP4")
  if int(args.get("data_parallel_size", 0)) != 1:
    raise RuntimeError("Gemma E4B P1 live engine must use DP1")
  runner = getattr(sampler, "_model_runner", None)
  if runner is None:
    raise RuntimeError("Gemma E4B P1 attestation has no live model runner")
  model_config = runner.model_config
  hf_config = model_config.hf_config
  text_config = getattr(hf_config, "text_config", hf_config)
  def config_value(name: str) -> Any:
    return getattr(text_config, name, getattr(hf_config, name, None))

  checks = {
      "layers": (int(text_config.num_hidden_layers), 42),
      "heads": (int(text_config.num_attention_heads), 8),
      "kv_heads": (model_config.get_total_num_kv_heads(), 2),
      "head_dim": (model_config.get_head_size(), 256),
      "hidden_size": (int(text_config.hidden_size), 2560),
      "intermediate_size": (int(text_config.intermediate_size), 10240),
      "vocab_size": (int(text_config.vocab_size), 262144),
      "sliding_window": (int(config_value("sliding_window")), 512),
      "final_logit_softcapping": (
          float(config_value("final_logit_softcapping")), 30.0
      ),
      "vocab_size_per_layer_input": (
          int(config_value("vocab_size_per_layer_input")), 262144
      ),
      "hidden_size_per_layer_input": (
          int(config_value("hidden_size_per_layer_input")), 256
      ),
      "num_kv_shared_layers": (
          int(config_value("num_kv_shared_layers")), 18
      ),
      "tie_word_embeddings": (
          bool(config_value("tie_word_embeddings")), True
      ),
      "attention_k_eq_v": (bool(config_value("attention_k_eq_v")), False),
      "enable_moe_block": (bool(config_value("enable_moe_block")), False),
      "layer_types": (
          tuple(config_value("layer_types")), EXPECTED_LAYER_TYPES
      ),
  }
  wrong = {key: pair for key, pair in checks.items() if pair[0] != pair[1]}
  if wrong:
    raise RuntimeError(f"Gemma E4B live architecture drifted: {wrong}")

  preprocess = sampler.config.mapping_config.preprocess_src_state
  if preprocess is None:
    raise RuntimeError("Gemma E4B P1 requires the registered state preprocessor")
  mapped_trainer_state = preprocess(trainer_state)

  # This helper is an arithmetic-free observer shared with the already-audited
  # exact engine-weight gate. All mapping inputs and preprocessing remain
  # Gemma-specific and mirror VllmSampler.update_params exactly.
  if attest_fn is None:
    from tunix.rl import canonical_qwen3_adapter  # pylint: disable=g-import-not-at-top
    attest_fn = canonical_qwen3_adapter.attest_exact_live_engine_weights

  result = attest_fn(
      sampler=sampler,
      trainer_state=mapped_trainer_state,
      tp_size=4,
  )
  if not result.get("equal"):
    raise RuntimeError(
        "Gemma E4B trainer/live-engine tensor mismatch: "
        f"indices={result.get('mismatch_indices')}"
    )
  mesh = dict(result.get("mesh_shape", ()))
  if mesh.get("data") != 1 or mesh.get("model") != 4:
    raise RuntimeError(f"Gemma E4B live engine mesh drifted: {mesh}")
  return {
      **result,
      "schema": "gemma4-e4b-p1-live-weight-attestation-v1",
      "model_family": "gemma4",
      "model_variant": "e4b-it",
      "architecture": {key: pair[0] for key, pair in checks.items()},
      "checkpoint_revision_sha256": hashlib.sha256(
          CHECKPOINT_REVISION.encode("ascii")
      ).hexdigest(),
  }
