# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Exact-token reconstruction and FrozenLake continuity receipts."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Mapping, Sequence
import uuid

import numpy as np


M15_TOKEN_CONTINUITY_ENV = "CANON_M15_TOKEN_CONTINUITY"
P57_TOKEN_CONTINUITY_ENV = "CANON_P57_TOKEN_CONTINUITY"
P57_TOKEN_CONTINUITY_DEBUG_ENV = "CANON_P57_TOKEN_CONTINUITY_DEBUG"
P57_TOKEN_CONTINUITY_DEBUG_FIRST_DIFF = "first-diff"
P57_TOKEN_CONTINUITY_DEBUG_COLLECT = "collect-64"
P57_TOKEN_CONTINUITY_DEBUG_RECORD_FULL = "record-full"
P57_TOKEN_CONTINUITY_DEBUG_VALUE = P57_TOKEN_CONTINUITY_DEBUG_FIRST_DIFF
P57_TITO_ONEHOST_NEUTRALITY_ENV = "CANON_P57_TITO_ONEHOST_NEUTRALITY"
P57_TITO_RUNNER_WITNESS_DIR_ENV = "CANON_P57_TITO_RUNNER_WITNESS_DIR"
P57_TOKEN_CONTINUITY_COLLECT_LIMIT = 64
P57_ACTOR_SNAPSHOT_THRESHOLDS = {
    "first-any": 0.0,
    "first-ge-1": 1.0,
    "first-ge-8": 8.0,
    "first-ge-32": 32.0,
}
_P57_TITO_GCS_PREFIX_RE = re.compile(
    r"gs://yuxzhang-tunix-models/canon-zero-tim/evidence/p57-tito/"
    r"[a-z0-9](?:[-a-z0-9]{0,62}[a-z0-9])?/attempt-(?:direct|[0-9]+)"
)

_COLLECTION_LOCK = threading.Lock()
_COLLECTION_STATE = {
    "active": False,
    "mode": "",
    "trajectories": 0,
    "compared_trajectories": 0,
    "unexercised_single_turn_trajectories": 0,
    "equal_trajectories": 0,
    "different_trajectories": 0,
    "later_turn_comparisons": 0,
    "engine_echo_comparisons": 0,
    "engine_echo_differences": 0,
    "token_difference_events": 0,
    "capsules_reserved": 0,
    "capsules_emitted": 0,
    "capsules_omitted": 0,
    "emission_failures": 0,
    "backward_transactions": 0,
    "gradient_microbatches": 0,
    "optimizer_commits": 0,
    "alignment_updates": 0,
}

_M15_FULL_IDENTITY = {
    "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
    "CANON_PROFILE_FILE": (
        "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-v1-hp.env"
    ),
    "CANON_PROFILE": "qwen3-8b-dp8-tp8-frozenlake-v1-hp",
    "CANON_V1_HP_FULL": "1",
    "CANON_P57_TIM_ARM": "zero",
    "CANON_P57_RUN_KIND": "train",
    "CANON_P57_EXPECTED_UPDATES": "300",
    "CANON_P57_STOP_AFTER_STEP": "300",
    "CANON_P57_WORKLOAD_CANDIDATE": "m15",
    "CANON_P57_DATA_SPLIT": "main",
    "CANON_P33_RUN_STAGE": "full",
    "CANON_P33_NO_COMMIT": "0",
    "CANON_P33_ENABLE_EVAL": "0",
    "CANON_P33_DISABLE_EVAL": "1",
    "CANON_P31_ENABLE_EVAL": "0",
    "CANON_FROZENLAKE_CKPT_MODE": "disabled",
    "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "1",
    "CANON_DP_SIZE": "8",
    "CANON_TP_SIZE": "8",
}
_P57_FULL_COMMON_IDENTITY = {
    name: expected
    for name, expected in _M15_FULL_IDENTITY.items()
    if name not in ("CANON_P57_WORKLOAD_CANDIDATE", "CANON_P57_DATA_SPLIT")
}
_P57_FULL_WORKLOAD_IDENTITIES = {
    "p45": {
        "CANON_P57_WORKLOAD_CANDIDATE": "",
        "CANON_P57_DATA_SPLIT": "",
    },
    "m15": {
        "CANON_P57_WORKLOAD_CANDIDATE": "m15",
        "CANON_P57_DATA_SPLIT": "main",
    },
}
_P57_TITO_ONEHOST_IDENTITY = {
    "CANON_V1_HP_FULL": "0",
    "CANON_P57_TIM_ARM": "zero",
    "CANON_P57_RUN_KIND": "train",
    "CANON_P57_EXPECTED_UPDATES": "3",
    "CANON_P57_STOP_AFTER_STEP": "3",
    "CANON_P57_WORKLOAD_CANDIDATE": "",
    "CANON_P57_DATA_SPLIT": "",
    "CANON_P33_RUN_STAGE": "full",
    "CANON_P33_NO_COMMIT": "0",
    "CANON_P33_ENABLE_EVAL": "0",
    "CANON_P33_DISABLE_EVAL": "1",
    "CANON_P31_ENABLE_EVAL": "0",
    "CANON_FROZENLAKE_CKPT_MODE": "disabled",
    "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
    "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
    "CANON_DP_SIZE": "1",
    "CANON_TP_SIZE": "4",
}
_P57_TITO_ONEHOST_ABSENT = (
    "CANON_P32_WORKLOAD",
    "CANON_PROFILE_FILE",
    "CANON_PROFILE",
    "CANON_P57_TITO_ROLLOUT_ONLY",
    P57_TITO_RUNNER_WITNESS_DIR_ENV,
)
_P57_TITO_DIAGNOSTIC_PROFILE = (
    "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-tito-diagnostic.env"
)
_P57_TITO_DIAGNOSTIC_COMMON_IDENTITY = {
    "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
    "CANON_PROFILE_FILE": _P57_TITO_DIAGNOSTIC_PROFILE,
    "CANON_PROFILE": "qwen3-8b-dp8-tp8-frozenlake-tito-diagnostic",
    "CANON_V1_HP_FULL": "0",
    "CANON_P57_RUN_KIND": "tito-diagnostic",
    "CANON_P57_TIM_ARM": "zero",
    "CANON_P57_EXPECTED_UPDATES": "1",
    "CANON_P57_STOP_AFTER_STEP": "1",
    "CANON_P33_RUN_STAGE": "rollout-only",
    "CANON_P33_NO_COMMIT": "1",
    "CANON_P33_ENABLE_EVAL": "0",
    "CANON_P33_DISABLE_EVAL": "1",
    "CANON_P31_ENABLE_EVAL": "0",
    "CANON_FROZENLAKE_CKPT_MODE": "disabled",
    "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
    "CANON_P57_TITO_ROLLOUT_ONLY": "1",
    "CANON_VLLM_ENABLE_PREFIX_CACHING": "0",
    "CANON_DP_SIZE": "8",
    "CANON_TP_SIZE": "8",
}

_M15_APC_DEBUG_PROFILE = (
    "cluster/profiles/qwen3-8b-dp8-tp8-frozenlake-apc-debug.env"
)
_M15_APC_DEBUG_IDENTITY = {
    "CANON_P32_WORKLOAD": "frozenlake-dp8-tp8",
    "CANON_PROFILE_FILE": _M15_APC_DEBUG_PROFILE,
    "CANON_PROFILE": "qwen3-8b-dp8-tp8-frozenlake-apc-debug",
    "CANON_V1_HP_FULL": "0",
    "CANON_P57_WORKLOAD_CANDIDATE": "m15",
    "CANON_P57_DATA_SPLIT": "main",
    "CANON_P33_RUN_STAGE": "backward-no-commit",
    "CANON_P33_NO_COMMIT": "1",
    "CANON_P38_PRECHECK_ONLY": "1",
    "CANON_P38_CONTROLLED_EXIT": "1",
    "CANON_P38_DIAGNOSTIC_ROUNDS": "3",
    "CANON_P38_DURABILITY_PROFILE": "m15-wide-v1",
    "CANON_P38_SEAM_OBSERVER": "layer",
    "CANON_P38_TAIL_OBSERVER": "1",
    "CANON_P33_ENABLE_EVAL": "0",
    "CANON_P33_DISABLE_EVAL": "1",
    "CANON_P31_ENABLE_EVAL": "0",
    "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
    "CANON_DP_SIZE": "8",
    "CANON_TP_SIZE": "8",
}
_M15_APC_DEBUG_ABSENT = (
    "CANON_P57_TIM_ARM",
    "CANON_P57_RUN_KIND",
    "CANON_P57_EXPECTED_UPDATES",
    "CANON_P57_STOP_AFTER_STEP",
    "CANON_FROZENLAKE_CKPT_MODE",
)
_M15_ONEHOST_IDENTITY = {
    "CANON_V1_HP_FULL": "0",
    "CANON_P57_WORKLOAD_CANDIDATE": "m15",
    "CANON_P57_DATA_SPLIT": "main",
    "CANON_P33_RUN_STAGE": "backward-no-commit",
    "CANON_P33_NO_COMMIT": "1",
    "CANON_P38_PRECHECK_ONLY": "1",
    "CANON_P38_CONTROLLED_EXIT": "1",
    "CANON_P38_DIAGNOSTIC_ROUNDS": "3",
    "CANON_P38_ONEHOST_REHEARSAL": "1",
    "CANON_P33_ENABLE_EVAL": "0",
    "CANON_P33_DISABLE_EVAL": "1",
    "CANON_P31_ENABLE_EVAL": "0",
    "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
    "CANON_DP_SIZE": "1",
    "CANON_TP_SIZE": "4",
}
_M15_ONEHOST_ABSENT = (
    "CANON_P32_WORKLOAD",
    "CANON_PROFILE_FILE",
    "CANON_PROFILE",
    "CANON_APC_M15_TARGET_DEBUG",
    "CANON_P57_TIM_ARM",
    "CANON_P57_RUN_KIND",
    "CANON_P57_EXPECTED_UPDATES",
    "CANON_P57_STOP_AFTER_STEP",
    "CANON_FROZENLAKE_CKPT_MODE",
    "CANON_P38_DURABILITY_PROFILE",
    "CANON_P38_SEAM_OBSERVER",
    "CANON_P38_TAIL_OBSERVER",
)


@dataclasses.dataclass(frozen=True, slots=True)
class FrozenLakeTokenContinuity:
  """One admitted FrozenLake token-continuity execution contract."""

  workload: str
  mode: str
  selector: str


@dataclasses.dataclass(frozen=True, slots=True)
class ContinuationPromptSegment:
  """One ordered, exact-token segment in a later-turn prompt."""

  kind: str
  turn_index: int
  done: bool | None
  tokens: np.ndarray


def _write_exclusive_json(path: Path, record: Mapping[str, Any]) -> tuple[Path, str, int]:
  """Writes one immutable mode-0600 JSON receipt without replacement."""
  payload = (
      json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n"
  ).encode("utf-8")
  path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
  path.parent.chmod(0o700)
  partial = path.with_name(
      f".{path.name}.partial-{os.getpid()}-{time.time_ns()}"
  )
  descriptor = os.open(
      partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
  )
  try:
    with os.fdopen(descriptor, "wb") as output:
      output.write(payload)
      output.flush()
      os.fsync(output.fileno())
    try:
      os.link(partial, path)
    except FileExistsError as error:
      raise FileExistsError(
          f"refusing a second immutable TiTO writer receipt: {path}"
      ) from error
    partial.unlink()
  except BaseException:
    partial.unlink(missing_ok=True)
    raise
  return path, hashlib.sha256(payload).hexdigest(), len(payload)


def write_tito_single_writer_receipt(
    values: Mapping[str, str] | None = None,
    *,
    state_dir: str | os.PathLike[str] | None = None,
) -> tuple[Path, str, int]:
  """Proves that exactly one Python controller owns record-full evidence."""
  env = os.environ if values is None else values
  if frozenlake_token_continuity_debug_mode(env) != (
      P57_TOKEN_CONTINUITY_DEBUG_RECORD_FULL
  ):
    raise ValueError("single-writer receipt requires record-full admission")
  contract = frozenlake_token_continuity(env)
  if contract is None:
    raise ValueError("single-writer receipt requires exact FrozenLake identity")
  root_value = state_dir if state_dir is not None else env.get("CANON_STATE")
  if not root_value or not os.path.isabs(os.fspath(root_value)):
    raise ValueError("single-writer receipt requires an absolute CANON_STATE")
  source_commit = env.get("CANON_EXPECT_COMMIT", "")
  image_identity = env.get("CANON_CLIENT_IMAGE", "")
  if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None or not image_identity:
    raise ValueError("single-writer receipt requires source and image identity")
  record = {
      "schema": "canon.p57-tito-single-writer.v1",
      "status": "PASS",
      "workload": contract.workload,
      "source_commit": source_commit,
      "image_identity": image_identity,
      "dp": int(env["CANON_DP_SIZE"]),
      "tp": int(env["CANON_TP_SIZE"]),
      "controller_pid": os.getpid(),
      "controller_hostname": os.uname().nodename,
      "writer_contract": "one-python-controller-o-excl",
      "neutrality_arm": env.get(P57_TITO_ONEHOST_NEUTRALITY_ENV),
  }
  path = Path(root_value) / "p57_tito_witness" / "single-writer.json"
  result = _write_exclusive_json(path, record)
  print(
      "[P57.TITO.SINGLE_WRITER] PASS "
      f"workload={contract.workload} dp={record['dp']} tp={record['tp']} "
      f"pid={record['controller_pid']} path={path}",
      flush=True,
  )
  return result


def begin_token_continuity_collection(
    values: Mapping[str, str] | None = None,
) -> None:
  """Starts the one process-wide bounded diagnostic collection."""
  mode = frozenlake_token_continuity_debug_mode(values)
  if mode not in (
      P57_TOKEN_CONTINUITY_DEBUG_COLLECT,
      P57_TOKEN_CONTINUITY_DEBUG_RECORD_FULL,
  ):
    raise ValueError("token collection requires collect-64 or record-full")
  with _COLLECTION_LOCK:
    if _COLLECTION_STATE["active"] or any(
        value
        for name, value in _COLLECTION_STATE.items()
        if name != "active"
    ):
      raise RuntimeError("P57 token collection state is not pristine")
    if mode == P57_TOKEN_CONTINUITY_DEBUG_RECORD_FULL:
      write_tito_single_writer_receipt(values=values)
    _COLLECTION_STATE["active"] = True
    _COLLECTION_STATE["mode"] = mode


def enforce_record_full_first_update_token_admission(
    rows: Sequence[Mapping[str, Any]],
    *,
    step: int,
    values: Mapping[str, str] | None = None,
) -> dict[str, Any] | None:
  """Reports update-zero token transport without stopping record-full."""
  env = os.environ if values is None else values
  if frozenlake_token_continuity_debug_mode(env) != (
      P57_TOKEN_CONTINUITY_DEBUG_RECORD_FULL
  ):
    return None
  if step != 0:
    return None
  different = sum(bool(row.get("token_different")) for row in rows)
  record = {
      "schema": "canon.p57-tito-first-update-token-observation.v2",
      "step": 0,
      "rows": len(rows),
      "different_rows": different,
      "verdict": "PASS" if different == 0 else "OBSERVED_DIFFERENT",
      "continue_training": True,
  }
  print(
      "[P57.TITO.FIRST_UPDATE_TOKEN_GATE] "
      f"{record['verdict']} step=0 rows={len(rows)} "
      f"different_rows={different} continue_training=1",
      flush=True,
  )
  return record


def run_tito_orbax_admission_probe(
    values: Mapping[str, str] | None = None,
    *,
    state_dir: str | os.PathLike[str] | None = None,
    manager_factory: Any = None,
    model_factory: Any = None,
    value_reader: Any = None,
) -> dict[str, Any] | None:
  """Proves the real record-full Orbax save/load path before rollout.

  The live GCS worker separately proves the gcloud transport. This probe uses
  the same Tunix CheckpointManager and Pathways persistence path as red-policy
  actor snapshots, then restores and byte-compares a tiny deterministic model.
  """
  env = os.environ if values is None else values
  if frozenlake_token_continuity_debug_mode(env) != (
      P57_TOKEN_CONTINUITY_DEBUG_RECORD_FULL
  ):
    return None
  if env.get(P57_TITO_ONEHOST_NEUTRALITY_ENV) == "on":
    return None
  contract = frozenlake_token_continuity(env)
  if (
      contract is None
      or env.get("CANON_DP_SIZE") != "8"
      or env.get("CANON_TP_SIZE") != "8"
  ):
    raise ValueError("Orbax admission probe requires production DP8xTP8")
  source_commit = env.get("CANON_EXPECT_COMMIT", "")
  image_identity = env.get("CANON_CLIENT_IMAGE", "")
  prefix = env.get("CANON_P57_TITO_GCS_PREFIX", "").rstrip("/")
  if (
      re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
      or not image_identity
      or _P57_TITO_GCS_PREFIX_RE.fullmatch(prefix) is None
  ):
    raise ValueError("Orbax admission probe source/image/GCS identity differs")
  root_value = state_dir if state_dir is not None else env.get("CANON_STATE")
  if not root_value or not os.path.isabs(os.fspath(root_value)):
    raise ValueError("Orbax admission probe requires an absolute CANON_STATE")
  receipt_path = Path(root_value) / "p57_tito_gcs" / "orbax-probe.json"
  probe_root = f"{prefix}/orbax-admission-probe"
  expected = np.asarray([57, 9, 3, 8], dtype=np.int32)

  if manager_factory is None:
    from tunix.sft import checkpoint_manager as checkpoint_manager_lib  # pylint: disable=g-import-not-at-top
    from tunix.sft import checkpoint_options as checkpoint_options_lib  # pylint: disable=g-import-not-at-top

    options = checkpoint_options_lib.TunixCheckpointingOptions(
        enable_async_checkpointing=False,
        save_on_close=False,
    )

    def manager_factory(value):
      return checkpoint_manager_lib.CheckpointManager(
          root_directory=value, options=options
      )

  if model_factory is None or value_reader is None:
    from flax import nnx  # pylint: disable=g-import-not-at-top
    import jax.numpy as jnp  # pylint: disable=g-import-not-at-top

    class _ProbeModel(nnx.Module):

      def __init__(self, value):
        self.payload = nnx.Param(jnp.asarray(value, dtype=jnp.int32))

    if model_factory is None:
      model_factory = _ProbeModel
    if value_reader is None:
      value_reader = lambda model: np.asarray(model.payload[...])

  manager = None
  failure_type = None
  restored_step = None
  restored_metadata: Any = None
  restored_equal = False
  started = time.perf_counter()
  try:
    manager = manager_factory(probe_root)
    if manager.latest_step() is not None:
      raise RuntimeError("Orbax admission probe root is not fresh")
    source_model = model_factory(expected)
    saved = manager.save(
        0,
        source_model,
        optimizer=None,
        force=True,
        custom_metadata={
            "schema": "canon.p57-tito-orbax-admission.v1",
            "source_commit": source_commit,
            "image_identity": image_identity,
            "workload": contract.workload,
            "dp": 8,
            "tp": 8,
        },
    )
    if saved is not True or manager.latest_step() != 0:
      raise RuntimeError("Orbax admission probe save did not close at step 0")
    restored_model = model_factory(np.zeros_like(expected))
    restored_step, restored_metadata = manager.maybe_restore(
        restored_model, step=0
    )
    restored = np.asarray(value_reader(restored_model), dtype=np.int32)
    restored_equal = bool(np.array_equal(restored, expected))
    if (
        restored_step != 0
        or not restored_equal
        or not isinstance(restored_metadata, Mapping)
        or restored_metadata.get("schema")
        != "canon.p57-tito-orbax-admission.v1"
        or restored_metadata.get("source_commit") != source_commit
        or restored_metadata.get("image_identity") != image_identity
    ):
      raise RuntimeError("Orbax admission probe restore differs")
  except Exception as exc:  # pylint: disable=broad-exception-caught
    failure_type = type(exc).__name__
  finally:
    if manager is not None:
      try:
        manager.close()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        failure_type = type(exc).__name__

  record = {
      "schema": "canon.p57-tito-orbax-admission-receipt.v1",
      "status": "PASS" if failure_type is None else "FAIL",
      "workload": contract.workload,
      "source_commit": source_commit,
      "image_identity": image_identity,
      "dp": 8,
      "tp": 8,
      "probe_root_sha256": hashlib.sha256(probe_root.encode()).hexdigest(),
      "saved_step": 0,
      "restored_step": restored_step,
      "restored_equal": restored_equal,
      "elapsed_seconds": time.perf_counter() - started,
      "failure_type": failure_type,
  }
  _write_exclusive_json(receipt_path, record)
  print(
      "[P57.TITO.ORBAX_PROBE] "
      f"{record['status']} workload={contract.workload} dp=8 tp=8 "
      f"restored_equal={int(restored_equal)} "
      f"root_sha256={record['probe_root_sha256']}",
      flush=True,
  )
  if failure_type is not None:
    raise RuntimeError(
        f"P57 TiTO Orbax admission probe failed: {failure_type}"
    )
  return record


def reserve_token_difference_capsule() -> int | None:
  """Records one diff and reserves its bounded evidence slot before I/O."""
  with _COLLECTION_LOCK:
    if not _COLLECTION_STATE["active"]:
      raise RuntimeError("P57 token collection is not active")
    if (
        _COLLECTION_STATE["capsules_reserved"]
        >= P57_TOKEN_CONTINUITY_COLLECT_LIMIT
    ):
      _COLLECTION_STATE["capsules_omitted"] += 1
      return None
    _COLLECTION_STATE["capsules_reserved"] += 1
    return int(_COLLECTION_STATE["capsules_reserved"])


def reserve_record_full_token_difference_event() -> int:
  """Reserves one unbounded, replay-complete record-full diff event."""
  with _COLLECTION_LOCK:
    if not _COLLECTION_STATE["active"]:
      raise RuntimeError("P57 token collection is not active")
    if _COLLECTION_STATE["mode"] != P57_TOKEN_CONTINUITY_DEBUG_RECORD_FULL:
      raise RuntimeError("record-full token event requires record-full mode")
    _COLLECTION_STATE["token_difference_events"] += 1
    _COLLECTION_STATE["capsules_reserved"] += 1
    if (
        _COLLECTION_STATE["token_difference_events"]
        != _COLLECTION_STATE["capsules_reserved"]
    ):
      raise RuntimeError("record-full token event accounting diverged")
    return int(_COLLECTION_STATE["token_difference_events"])


def record_token_capsule_emission(*, succeeded: bool) -> None:
  """Accounts for one already-reserved capsule emission attempt."""
  with _COLLECTION_LOCK:
    if not _COLLECTION_STATE["active"]:
      raise RuntimeError("P57 token collection is not active")
    if succeeded:
      _COLLECTION_STATE["capsules_emitted"] += 1
    else:
      _COLLECTION_STATE["emission_failures"] += 1


def record_token_collection_trajectory(
    *, different: bool, later_turns: int
) -> None:
  """Accounts for one completed independent diagnostic trajectory."""
  if not isinstance(later_turns, int) or later_turns < 0:
    raise ValueError("later_turns must be a nonnegative integer")
  with _COLLECTION_LOCK:
    if not _COLLECTION_STATE["active"]:
      raise RuntimeError("P57 token collection is not active")
    _COLLECTION_STATE["trajectories"] += 1
    _COLLECTION_STATE["later_turn_comparisons"] += later_turns
    if later_turns == 0 and not different:
      _COLLECTION_STATE["unexercised_single_turn_trajectories"] += 1
    else:
      _COLLECTION_STATE["compared_trajectories"] += 1
      key = "different_trajectories" if different else "equal_trajectories"
      _COLLECTION_STATE[key] += 1


def record_prompt_echo_comparison(*, equal: bool) -> None:
  """Accounts for one attributable submitted/RequestOutput comparison."""
  with _COLLECTION_LOCK:
    if not _COLLECTION_STATE["active"]:
      raise RuntimeError("P57 token collection is not active")
    _COLLECTION_STATE["engine_echo_comparisons"] += 1
    if not equal:
      _COLLECTION_STATE["engine_echo_differences"] += 1


def record_full_update(update: Mapping[str, Any]) -> None:
  """Accounts from one validated real segmented-update receipt."""
  with _COLLECTION_LOCK:
    if not _COLLECTION_STATE["active"]:
      return
    if _COLLECTION_STATE["mode"] != P57_TOKEN_CONTINUITY_DEBUG_RECORD_FULL:
      return
    if update.get("verdict") != "PASS":
      raise RuntimeError("record-full cannot account a failed update")
    microsteps = update.get("microsteps")
    commits = update.get("commits")
    hashes = update.get("alignment_hashes")
    if (
        type(microsteps) is not int
        or microsteps <= 0
        or commits != 1
        or not isinstance(hashes, list)
        or not hashes
    ):
      raise RuntimeError("record-full update receipt is incomplete")
    _COLLECTION_STATE["backward_transactions"] += 1
    _COLLECTION_STATE["gradient_microbatches"] += microsteps
    _COLLECTION_STATE["optimizer_commits"] += commits
    _COLLECTION_STATE["alignment_updates"] += len(hashes)


def token_collection_snapshot() -> dict[str, int | bool]:
  """Returns an immutable copy of the process-wide collection counters."""
  with _COLLECTION_LOCK:
    return dict(_COLLECTION_STATE)


def _reset_token_collection_for_test() -> None:
  with _COLLECTION_LOCK:
    for name in _COLLECTION_STATE:
      _COLLECTION_STATE[name] = False if name == "active" else "" if name == "mode" else 0


def m15_token_continuity_mode(
    values: Mapping[str, str] | None = None,
) -> str | None:
  """Returns the admitted M15 continuity mode, failing closed on drift."""
  env = os.environ if values is None else values
  if (
      M15_TOKEN_CONTINUITY_ENV in env
      and P57_TOKEN_CONTINUITY_ENV in env
  ):
    raise ValueError(
        "CANON_M15_TOKEN_CONTINUITY and CANON_P57_TOKEN_CONTINUITY are "
        "mutually exclusive"
    )
  if M15_TOKEN_CONTINUITY_ENV not in env:
    return None
  mode = env[M15_TOKEN_CONTINUITY_ENV]
  if mode not in ("verify", "exact"):
    raise ValueError(
        "CANON_M15_TOKEN_CONTINUITY must be absent, 'verify', or 'exact'"
    )
  onehost_identity = env.get("CANON_P38_ONEHOST_REHEARSAL") == "1"
  debug_identity = env.get("CANON_PROFILE_FILE") == _M15_APC_DEBUG_PROFILE
  identity = (
      _M15_ONEHOST_IDENTITY
      if onehost_identity
      else _M15_APC_DEBUG_IDENTITY
      if debug_identity
      else _M15_FULL_IDENTITY
  )
  drift = {
      name: (env.get(name), expected)
      for name, expected in identity.items()
      if env.get(name) != expected
  }
  if onehost_identity:
    apc = env.get("CANON_VLLM_ENABLE_PREFIX_CACHING")
    admitted_apc = ("0", "1") if mode == "exact" else ("0",)
    if apc not in admitted_apc:
      drift["CANON_VLLM_ENABLE_PREFIX_CACHING"] = (
          apc,
          "|".join(admitted_apc),
      )
    for name in _M15_ONEHOST_ABSENT:
      if env.get(name) not in (None, ""):
        drift[name] = (env.get(name), "absent")
  elif debug_identity:
    if mode != "exact":
      raise ValueError("M15 APC debug admits exact token continuity only")
    arm = env.get("CANON_APC_M15_TARGET_DEBUG")
    if arm not in ("off", "on"):
      drift["CANON_APC_M15_TARGET_DEBUG"] = (arm, "off|on")
    for name in _M15_APC_DEBUG_ABSENT:
      if env.get(name) not in (None, ""):
        drift[name] = (env.get(name), "absent")
  elif mode != "exact":
    raise ValueError("M15 full training admits exact continuity only")
  if drift:
    details = ", ".join(
        f"{name}={actual!r} expected {expected!r}"
        for name, (actual, expected) in sorted(drift.items())
    )
    raise ValueError(
        f"M15 token-continuity {mode} is outside its registered identity: "
        + details
    )
  forbidden_checkpoint_values = "".join(
      env.get(name, "")
      for name in (
          "CANON_FROZENLAKE_CKPT_ROOT",
          "CANON_FROZENLAKE_CKPT_TAG",
          "CANON_FROZENLAKE_CKPT_INTERVAL",
          "CANON_FROZENLAKE_CKPT_MAX_TO_KEEP",
          "CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL",
      )
  )
  if forbidden_checkpoint_values:
    raise ValueError(
        f"M15 token-continuity {mode} requires its checkpoint-free "
        "registered run identity"
    )
  return mode


def frozenlake_token_continuity(
    values: Mapping[str, str] | None = None,
) -> FrozenLakeTokenContinuity | None:
  """Returns the admitted legacy or P57 FrozenLake continuity contract."""
  env = os.environ if values is None else values
  if (
      M15_TOKEN_CONTINUITY_ENV in env
      and P57_TOKEN_CONTINUITY_ENV in env
  ):
    raise ValueError(
        "CANON_M15_TOKEN_CONTINUITY and CANON_P57_TOKEN_CONTINUITY are "
        "mutually exclusive"
    )
  if P57_TOKEN_CONTINUITY_ENV not in env:
    if P57_TITO_ONEHOST_NEUTRALITY_ENV in env:
      raise ValueError(
          "P57 TiTO one-host neutrality requires generic exact continuity"
      )
    legacy_mode = m15_token_continuity_mode(env)
    if legacy_mode is None:
      return None
    return FrozenLakeTokenContinuity(
        workload="m15",
        mode=legacy_mode,
        selector=M15_TOKEN_CONTINUITY_ENV,
    )

  mode = env[P57_TOKEN_CONTINUITY_ENV]
  if mode != "exact":
    raise ValueError(
        "CANON_P57_TOKEN_CONTINUITY must be absent or exactly 'exact'"
    )
  workload_identity = (
      env.get("CANON_P57_WORKLOAD_CANDIDATE", ""),
      env.get("CANON_P57_DATA_SPLIT", ""),
  )
  if workload_identity == ("", ""):
    workload = "p45"
  elif workload_identity == ("m15", "main"):
    workload = "m15"
  else:
    raise ValueError(
        "P57 exact token continuity requires P45 readiness or M15/main, got "
        f"candidate={workload_identity[0]!r} split={workload_identity[1]!r}"
    )
  onehost_neutrality = env.get(P57_TITO_ONEHOST_NEUTRALITY_ENV)
  if onehost_neutrality is not None and onehost_neutrality not in ("off", "on"):
    raise ValueError(
        "CANON_P57_TITO_ONEHOST_NEUTRALITY must be absent, 'off', or 'on'"
    )
  if onehost_neutrality is not None and workload != "p45":
    raise ValueError("P57 TiTO one-host neutrality admits P45 only")
  diagnostic = (
      env.get(P57_TOKEN_CONTINUITY_DEBUG_ENV)
      == P57_TOKEN_CONTINUITY_DEBUG_COLLECT
  )
  if onehost_neutrality is not None:
    identity = _P57_TITO_ONEHOST_IDENTITY
  else:
    identity = {
        **(
            _P57_TITO_DIAGNOSTIC_COMMON_IDENTITY
            if diagnostic
            else _P57_FULL_COMMON_IDENTITY
        ),
        **_P57_FULL_WORKLOAD_IDENTITIES[workload],
    }
  drift = {
      name: (env.get(name), expected)
      for name, expected in identity.items()
      if env.get(name, "") != expected
  }
  if drift:
    details = ", ".join(
        f"{name}={actual!r} expected {expected!r}"
        for name, (actual, expected) in sorted(drift.items())
    )
    raise ValueError(
        f"P57 {workload} token-continuity exact is outside its registered "
        f"identity: {details}"
    )
  if onehost_neutrality is not None:
    expected_debug = (
        P57_TOKEN_CONTINUITY_DEBUG_RECORD_FULL
        if onehost_neutrality == "on"
        else None
    )
    actual_debug = env.get(P57_TOKEN_CONTINUITY_DEBUG_ENV)
    if actual_debug != expected_debug:
      raise ValueError(
          "P57 TiTO one-host neutrality debug identity differs: "
          f"arm={onehost_neutrality!r} debug={actual_debug!r} "
          f"expected={expected_debug!r}"
      )
    for name in _P57_TITO_ONEHOST_ABSENT:
      if env.get(name) not in (None, ""):
        raise ValueError(
            "P57 TiTO one-host neutrality requires "
            f"{name} absent, got {env.get(name)!r}"
        )
  forbidden_checkpoint_values = "".join(
      env.get(name, "")
      for name in (
          "CANON_FROZENLAKE_CKPT_ROOT",
          "CANON_FROZENLAKE_CKPT_TAG",
          "CANON_FROZENLAKE_CKPT_INTERVAL",
          "CANON_FROZENLAKE_CKPT_MAX_TO_KEEP",
          "CANON_FROZENLAKE_CKPT_MILESTONE_INTERVAL",
      )
  )
  if forbidden_checkpoint_values:
    raise ValueError(
        f"P57 {workload} exact token continuity requires its "
        "checkpoint-free registered run identity"
    )
  return FrozenLakeTokenContinuity(
      workload=workload,
      mode=mode,
      selector=P57_TOKEN_CONTINUITY_ENV,
  )


def frozenlake_token_continuity_debug_enabled(
    values: Mapping[str, str] | None = None,
) -> bool:
  """Returns whether a bounded P57 token diagnostic is admitted."""
  return frozenlake_token_continuity_debug_mode(values) is not None


def frozenlake_token_continuity_debug_mode(
    values: Mapping[str, str] | None = None,
) -> str | None:
  """Returns the closed P57 diagnostic mode, failing closed on identity drift."""
  env = os.environ if values is None else values
  if P57_TOKEN_CONTINUITY_DEBUG_ENV not in env:
    if env.get(P57_TITO_ONEHOST_NEUTRALITY_ENV) == "on":
      raise ValueError(
          "P57 TiTO one-host neutrality on requires record-full debug"
      )
    return None
  value = env[P57_TOKEN_CONTINUITY_DEBUG_ENV]
  if value not in (
      P57_TOKEN_CONTINUITY_DEBUG_FIRST_DIFF,
      P57_TOKEN_CONTINUITY_DEBUG_COLLECT,
      P57_TOKEN_CONTINUITY_DEBUG_RECORD_FULL,
  ):
    raise ValueError(
        "CANON_P57_TOKEN_CONTINUITY_DEBUG must be absent, 'first-diff', or "
        "'collect-64', or 'record-full'"
    )
  contract = frozenlake_token_continuity(env)
  if (
      contract is None
      or contract.mode != "exact"
      or contract.selector != P57_TOKEN_CONTINUITY_ENV
  ):
    raise ValueError(
        "P57 token-continuity debug requires a generic exact P45/M15 "
        "admission"
    )
  witness_dir = env.get(P57_TITO_RUNNER_WITNESS_DIR_ENV)
  onehost_neutrality = env.get(P57_TITO_ONEHOST_NEUTRALITY_ENV)
  if onehost_neutrality is not None and (
      onehost_neutrality != "on"
      or value != P57_TOKEN_CONTINUITY_DEBUG_RECORD_FULL
  ):
    raise ValueError(
        "P57 TiTO one-host neutrality only admits arm=on with record-full"
    )
  if value in (
      P57_TOKEN_CONTINUITY_DEBUG_FIRST_DIFF,
      P57_TOKEN_CONTINUITY_DEBUG_RECORD_FULL,
  ):
    if witness_dir not in (None, ""):
      raise ValueError(f"{value} diagnostics forbid the runner witness")
    if value == P57_TOKEN_CONTINUITY_DEBUG_RECORD_FULL:
      state_dir = env.get("CANON_STATE")
      if not state_dir or not os.path.isabs(state_dir):
        raise ValueError("record-full requires an absolute CANON_STATE")
  else:
    state_dir = env.get("CANON_STATE")
    if not state_dir or not os.path.isabs(state_dir):
      raise ValueError("collect-64 requires an absolute CANON_STATE")
    expected = os.path.realpath(
        os.path.join(state_dir, "p57_tito_witness", "runner")
    )
    if not witness_dir or os.path.realpath(witness_dir) != expected:
      raise ValueError(
          "collect-64 requires its registered runner witness directory"
      )
  return value


def _integer_vector(value: Any, *, field: str) -> np.ndarray:
  array = np.asarray(value)
  if array.ndim != 1 or array.dtype.kind not in "iu":
    raise TypeError(f"{field} must be a 1-D integer array")
  if np.any(array < 0):
    raise ValueError(f"{field} contains a negative token id")
  if np.any(array > np.iinfo(np.int32).max):
    raise ValueError(f"{field} contains a token id outside int32")
  return np.asarray(array, dtype=np.int32)


def continuation_prompt_segments(
    trajectory: Any,
    *,
    contract: str,
) -> tuple[ContinuationPromptSegment, ...]:
  """Returns the canonical ordered ledger for a later-turn exact prompt."""
  if not trajectory.steps:
    raise RuntimeError(f"{contract} token continuity requires a completed turn")

  raw_prompt = _integer_vector(
      getattr(trajectory, "prompt_tokens", None),
      field=f"{contract} trajectory prompt tokens",
  )
  prompt_length = getattr(trajectory, "prompt_length", None)
  if (
      not isinstance(prompt_length, (int, np.integer))
      or not 0 < int(prompt_length) <= raw_prompt.size
  ):
    raise ValueError(
        f"{contract} trajectory prompt length is absent or outside its token "
        "width"
    )

  segments = [
      ContinuationPromptSegment(
          kind="initial_prompt",
          turn_index=-1,
          done=None,
          tokens=raw_prompt[-int(prompt_length):],
      )
  ]
  for step_index, step in enumerate(trajectory.steps):
    assistant_tokens = getattr(step, "assistant_tokens", None)
    if assistant_tokens is None:
      raise ValueError(
          f"{contract} turn {step_index} has no exact sampled assistant tokens"
      )
    done = bool(getattr(step, "done", False))
    segments.append(
        ContinuationPromptSegment(
            kind="assistant",
            turn_index=step_index,
            done=done,
            tokens=_integer_vector(
                assistant_tokens,
                field=f"{contract} turn {step_index} assistant tokens",
            ),
        )
    )

    env_tokens = getattr(step, "env_tokens", None)
    if env_tokens is None:
      if not done:
        raise ValueError(
            f"{contract} nonterminal turn {step_index} has no environment "
            "tokens"
        )
      continue
    segments.append(
        ContinuationPromptSegment(
            kind="environment",
            turn_index=step_index,
            done=done,
            tokens=_integer_vector(
                env_tokens,
                field=f"{contract} turn {step_index} environment tokens",
            ),
        )
    )
  return tuple(segments)


def reconstruct_continuation_prompt_tokens(
    trajectory: Any,
    response_token_count: int,
    *,
    contract: str,
) -> np.ndarray:
  """Reconstructs the exact token stream sampled across completed turns."""
  if (
      not isinstance(response_token_count, (int, np.integer))
      or int(response_token_count) < 0
  ):
    raise ValueError(f"{contract} response token counter must be nonnegative")

  segments = continuation_prompt_segments(trajectory, contract=contract)
  prompt_token_ids = np.concatenate(
      [segment.tokens for segment in segments], axis=0
  )
  initial_prompt_length = int(segments[0].tokens.size)
  expected = initial_prompt_length + int(response_token_count)
  if prompt_token_ids.size != expected:
    raise ValueError(
        f"{contract} exact prompt width differs from the trajectory response "
        f"counter: {prompt_token_ids.size} vs {expected}"
    )
  return prompt_token_ids


def trainer_bc_prompt_prefix(
    prompt_tokens: Sequence[int] | np.ndarray,
    prompt_length: int,
    conversation_tokens: Sequence[int] | np.ndarray,
    completed_response_token_count: int,
    *,
    contract: str,
) -> np.ndarray:
  """Builds the exact `prompt + completion-prefix` consumed by B/C."""
  raw_prompt = _integer_vector(
      prompt_tokens, field=f"{contract} trainer prompt tokens"
  )
  conversation = _integer_vector(
      conversation_tokens, field=f"{contract} trainer conversation tokens"
  )
  if (
      not isinstance(prompt_length, (int, np.integer))
      or not 0 < int(prompt_length) <= raw_prompt.size
  ):
    raise ValueError(f"{contract} trainer prompt length is invalid")
  if (
      not isinstance(completed_response_token_count, (int, np.integer))
      or not 0
      <= int(completed_response_token_count)
      <= conversation.size
  ):
    raise ValueError(
        f"{contract} completed response width is outside the trainer "
        "conversation"
    )
  return np.concatenate(
      (
          raw_prompt[-int(prompt_length):],
          conversation[: int(completed_response_token_count)],
      ),
      axis=0,
  )


def unpadded_rollout_prompt_tokens(rollout_output: Any) -> np.ndarray:
  """Extracts the single prompt actually consumed by the rollout worker."""
  raw_prompts = np.asarray(rollout_output.left_padded_prompt_tokens)
  lengths = np.asarray(rollout_output.prompt_lengths)
  if raw_prompts.ndim != 2 or raw_prompts.shape[0] != 1:
    raise ValueError(
        "FrozenLake token observer expected one 2-D left-padded prompt, got "
        f"{raw_prompts.shape}"
    )
  if raw_prompts.dtype.kind not in "iu":
    raise TypeError("FrozenLake rollout prompt tokens must be integers")
  if lengths.shape != (1,) or lengths.dtype.kind not in "iu":
    raise ValueError(
        "FrozenLake token observer expected one integer prompt length, got "
        f"shape={lengths.shape} dtype={lengths.dtype}"
    )
  prompt_length = int(lengths[0])
  if not 0 < prompt_length <= raw_prompts.shape[1]:
    raise ValueError(
        "FrozenLake rollout prompt length is outside its padded token width: "
        f"{prompt_length} vs {raw_prompts.shape[1]}"
    )
  return _integer_vector(
      raw_prompts[0, -prompt_length:],
      field="FrozenLake rollout prompt tokens",
  )


def _digest(tokens: np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(tokens).tobytes()).hexdigest()


def _prompt_witness_digest(tokens: np.ndarray) -> str:
  """Matches vLLM's portable little-endian int64 prompt witness hash."""
  portable = np.ascontiguousarray(np.asarray(tokens, dtype="<i8"))
  return hashlib.sha256(portable.tobytes()).hexdigest()


def continuity_debug_receipts(
    trajectory: Any,
    actual: Sequence[int] | np.ndarray,
    expected: Sequence[int] | np.ndarray,
    *,
    turn: int,
    workload: str,
    trajectory_id: str | None = None,
    policy_step: Any = None,
    pair_index: Any = None,
    group_id: Any = None,
    request_id: str | None = None,
    event_index: int | None = None,
    chunk_size: int = 256,
) -> tuple[str, ...]:
  """Builds a complete, chunked token ledger for one exact first mismatch."""
  if workload not in ("p45", "m15"):
    raise ValueError(f"unsupported FrozenLake debug workload: {workload!r}")
  if not isinstance(turn, (int, np.integer)) or int(turn) <= 0:
    raise ValueError("FrozenLake debug turn must be a positive integer")
  if not isinstance(chunk_size, int) or chunk_size <= 0:
    raise ValueError("FrozenLake debug chunk size must be positive")
  if trajectory_id is not None and (
      len(trajectory_id) != 32
      or any(character not in "0123456789abcdef" for character in trajectory_id)
  ):
    raise ValueError("FrozenLake debug trajectory_id must be 32 lowercase hex")
  if request_id is not None and (not isinstance(request_id, str) or not request_id):
    raise ValueError("FrozenLake debug request_id must be a nonempty string")
  if event_index is not None and (
      type(event_index) is not int or event_index <= 0
  ):
    raise ValueError("FrozenLake debug event_index must be a positive integer")
  actual_tokens = _integer_vector(
      actual, field=f"{workload.upper()} debug actual prompt tokens"
  )
  expected_tokens = _integer_vector(
      expected, field=f"{workload.upper()} debug expected prompt tokens"
  )
  common = min(actual_tokens.size, expected_tokens.size)
  unequal = np.flatnonzero(actual_tokens[:common] != expected_tokens[:common])
  if unequal.size:
    first_mismatch = int(unequal[0])
  elif actual_tokens.size != expected_tokens.size:
    first_mismatch = common
  else:
    raise ValueError("first-diff diagnostics require unequal token streams")
  capsule_id = uuid.uuid4().hex

  segments = continuation_prompt_segments(
      trajectory, contract=f"{workload.upper()} debug"
  )
  reconstructed = np.concatenate(
      [segment.tokens for segment in segments], axis=0
  )
  if not np.array_equal(reconstructed, expected_tokens):
    raise ValueError(
        "FrozenLake debug segment ledger does not reconstruct expected tokens"
    )

  records: list[dict[str, Any]] = []

  def _append_chunks(
      *,
      stream: str,
      segment_index: int,
      kind: str,
      turn_index: int,
      done: bool | None,
      tokens: np.ndarray,
  ) -> None:
    segment_sha = _digest(tokens)
    offsets = (
        (0,)
        if tokens.size == 0
        else range(0, int(tokens.size), chunk_size)
    )
    for chunk_index, offset in enumerate(offsets):
      chunk = tokens[offset:offset + chunk_size]
      record: dict[str, Any] = {
          "schema": "p57-token-first-diff-v1",
          "record": "token_chunk",
          "capsule_id": capsule_id,
          "workload": workload,
          "stream": stream,
          "segment_index": segment_index,
          "kind": kind,
          "turn_index": turn_index,
          "chunk_index": chunk_index,
          "offset": offset,
          "length": int(chunk.size),
          "segment_length": int(tokens.size),
          "segment_sha256": segment_sha,
          "chunk_sha256": _digest(chunk),
          "tokens": [int(token) for token in chunk],
      }
      if done is not None:
        record["done"] = done
      records.append(record)

  _append_chunks(
      stream="actual",
      segment_index=0,
      kind="serving_prompt",
      turn_index=int(turn),
      done=None,
      tokens=actual_tokens,
  )
  for segment_index, segment in enumerate(segments):
    _append_chunks(
        stream="expected",
        segment_index=segment_index,
        kind=segment.kind,
        turn_index=segment.turn_index,
        done=segment.done,
        tokens=segment.tokens,
    )

  header = {
      "schema": "p57-token-first-diff-v1",
      "record": "header",
      "capsule_id": capsule_id,
      "workload": workload,
      "trajectory_id": trajectory_id,
      "policy_step": None if policy_step is None else int(policy_step),
      "turn": int(turn),
      "pair_index": None if pair_index is None else str(pair_index),
      "group_id": None if group_id is None else str(group_id),
      "trajectory_steps": len(trajectory.steps),
      "first_mismatch": first_mismatch,
      "actual_tokens": int(actual_tokens.size),
      "expected_tokens": int(expected_tokens.size),
      "actual_sha256": _digest(actual_tokens),
      "expected_sha256": _digest(expected_tokens),
      "segments": len(segments),
      "token_chunk_records": len(records),
      "records_metadata_sha256": _debug_records_metadata_digest(records),
  }
  if request_id is not None:
    header["request_id"] = request_id
  if event_index is not None:
    header["event_index"] = event_index
  compact = {"sort_keys": True, "separators": (",", ":")}
  lines = [
      "[CANON_P57_TOKEN_CONTINUITY_DEBUG] "
      + json.dumps(header, **compact)
  ]
  lines.extend(
      "[CANON_P57_TOKEN_CONTINUITY_DEBUG_JSON] "
      + json.dumps(record, **compact)
      for record in records
  )
  return tuple(lines)


_DEBUG_HEADER_PREFIX = "[CANON_P57_TOKEN_CONTINUITY_DEBUG] "
_DEBUG_CHUNK_PREFIX = "[CANON_P57_TOKEN_CONTINUITY_DEBUG_JSON] "


def _debug_records_metadata_digest(records: Sequence[Mapping[str, Any]]) -> str:
  """Hashes all non-token chunk metadata in canonical segment order."""
  ordered = sorted(
      records,
      key=lambda record: (
          record["stream"],
          record["segment_index"],
          record["chunk_index"],
      ),
  )
  metadata = [
      {name: value for name, value in record.items() if name != "tokens"}
      for record in ordered
  ]
  payload = json.dumps(
      metadata, sort_keys=True, separators=(",", ":")
  ).encode("utf-8")
  return hashlib.sha256(payload).hexdigest()


def debug_capsule_from_receipts(
    lines: Sequence[str],
    *,
    capsule_id: str | None = None,
) -> dict[str, Any]:
  """Validates and expands one replay capsule from interleavable log lines."""
  headers: dict[str, dict[str, Any]] = {}
  chunks: dict[str, list[dict[str, Any]]] = {}
  for raw_line in lines:
    line = raw_line.strip()
    prefix = None
    if line.startswith(_DEBUG_HEADER_PREFIX):
      prefix = _DEBUG_HEADER_PREFIX
      expected_record = "header"
    elif line.startswith(_DEBUG_CHUNK_PREFIX):
      prefix = _DEBUG_CHUNK_PREFIX
      expected_record = "token_chunk"
    else:
      continue
    record = json.loads(line.removeprefix(prefix))
    if (
        not isinstance(record, dict)
        or record.get("schema") != "p57-token-first-diff-v1"
        or record.get("record") != expected_record
    ):
      raise ValueError("invalid P57 token first-diff log record")
    record_id = record.get("capsule_id")
    if not isinstance(record_id, str) or len(record_id) != 32:
      raise ValueError("invalid P57 token first-diff capsule id")
    if expected_record == "header":
      if record_id in headers:
        raise ValueError(f"duplicate first-diff header for {record_id}")
      headers[record_id] = record
    else:
      chunks.setdefault(record_id, []).append(record)

  if capsule_id is None:
    if len(headers) != 1:
      raise ValueError(
          "first-diff log must contain exactly one header unless capsule_id "
          "is selected"
      )
    capsule_id = next(iter(headers))
  if capsule_id not in headers:
    raise ValueError(f"first-diff header is missing for {capsule_id}")
  header = headers[capsule_id]
  if header.get("workload") not in ("p45", "m15"):
    raise ValueError("first-diff header has invalid workload")
  integer_header_fields = (
      "turn",
      "trajectory_steps",
      "first_mismatch",
      "actual_tokens",
      "expected_tokens",
      "segments",
      "token_chunk_records",
  )
  if any(type(header.get(name)) is not int for name in integer_header_fields):
    raise ValueError("first-diff header has invalid integer metadata")
  if (
      header["turn"] < 1
      or header["trajectory_steps"] != header["turn"]
      or header["first_mismatch"] < 0
      or header["actual_tokens"] < 0
      or header["expected_tokens"] < 0
      or header["segments"] < 1
      or header["token_chunk_records"] < 2
  ):
    raise ValueError("first-diff header has inconsistent shape metadata")
  for name in (
      "actual_sha256",
      "expected_sha256",
      "records_metadata_sha256",
  ):
    value = header.get(name)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
      raise ValueError(f"first-diff header has invalid {name}")
  records = chunks.get(capsule_id, [])
  if len(records) != header.get("token_chunk_records"):
    raise ValueError(
        f"first-diff chunk count differs for {capsule_id}: "
        f"{len(records)} vs {header.get('token_chunk_records')!r}"
    )

  grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
  for record in records:
    key = (record.get("stream"), record.get("segment_index"))
    if (
        key[0] not in ("actual", "expected")
        or type(key[1]) is not int
        or key[1] < 0
        or type(record.get("chunk_index")) is not int
        or record["chunk_index"] < 0
        or type(record.get("offset")) is not int
        or record["offset"] < 0
        or type(record.get("length")) is not int
        or record["length"] < 0
        or type(record.get("segment_length")) is not int
        or record["segment_length"] < 0
        or not isinstance(record.get("kind"), str)
        or type(record.get("turn_index")) is not int
    ):
      raise ValueError("first-diff chunk has invalid stream/segment")
    if record.get("workload") != header["workload"]:
      raise ValueError("first-diff chunk workload differs from its header")
    for name in ("segment_sha256", "chunk_sha256"):
      value = record.get(name)
      if (
          not isinstance(value, str)
          or len(value) != 64
          or any(character not in "0123456789abcdef" for character in value)
      ):
        raise ValueError(f"first-diff chunk has invalid {name}")
    grouped.setdefault(key, []).append(record)
  if _debug_records_metadata_digest(records) != header["records_metadata_sha256"]:
    raise ValueError("first-diff chunk metadata digest differs")

  expected_keys = {
      ("actual", 0),
      *(("expected", index) for index in range(header["segments"])),
  }
  if set(grouped) != expected_keys:
    raise ValueError("first-diff stream/segment topology differs")

  def _assemble(key: tuple[str, int]) -> tuple[dict[str, Any], np.ndarray]:
    ordered = sorted(grouped.get(key, []), key=lambda item: item["chunk_index"])
    if not ordered:
      raise ValueError(f"first-diff segment is missing: {key}")
    tokens: list[int] = []
    next_offset = 0
    segment_length = ordered[0].get("segment_length")
    segment_sha = ordered[0].get("segment_sha256")
    stable_metadata = {
        name: ordered[0].get(name)
        for name in (
            "workload",
            "stream",
            "segment_index",
            "kind",
            "turn_index",
        )
    }
    stable_metadata["done"] = ordered[0].get("done", None)
    stable_metadata["has_done"] = "done" in ordered[0]
    for chunk_index, record in enumerate(ordered):
      if record.get("chunk_index") != chunk_index:
        raise ValueError(f"first-diff chunk order is not contiguous: {key}")
      if record.get("offset") != next_offset:
        raise ValueError(f"first-diff chunk offset is not contiguous: {key}")
      if (
          record.get("segment_length") != segment_length
          or record.get("segment_sha256") != segment_sha
      ):
        raise ValueError(f"first-diff segment metadata drifted: {key}")
      current_metadata = {
          name: record.get(name)
          for name in (
              "workload",
              "stream",
              "segment_index",
              "kind",
              "turn_index",
          )
      }
      current_metadata["done"] = record.get("done", None)
      current_metadata["has_done"] = "done" in record
      if current_metadata != stable_metadata:
        raise ValueError(f"first-diff segment attribution drifted: {key}")
      raw_tokens = record.get("tokens")
      chunk = (
          np.asarray([], dtype=np.int32)
          if raw_tokens == []
          else _integer_vector(
              raw_tokens, field=f"first-diff {key} chunk {chunk_index}"
          )
      )
      if int(chunk.size) != record.get("length"):
        raise ValueError(f"first-diff chunk length differs: {key}")
      if _digest(chunk) != record.get("chunk_sha256"):
        raise ValueError(f"first-diff chunk hash differs: {key}")
      tokens.extend(int(token) for token in chunk)
      next_offset += int(chunk.size)
    assembled = np.asarray(tokens, dtype=np.int32)
    if assembled.size != segment_length or _digest(assembled) != segment_sha:
      raise ValueError(f"first-diff segment hash/length differs: {key}")
    metadata = {
        name: ordered[0].get(name)
        for name in ("stream", "segment_index", "kind", "turn_index", "done")
        if name in ordered[0]
    }
    metadata.update({
        "length": int(assembled.size),
        "sha256": _digest(assembled),
        "tokens": tokens,
    })
    return metadata, assembled

  actual_metadata, actual = _assemble(("actual", 0))
  expected_indices = sorted(
      segment_index
      for stream, segment_index in grouped
      if stream == "expected"
  )
  if expected_indices != list(range(header.get("segments", -1))):
    raise ValueError("first-diff expected segment indices are not contiguous")
  expected_metadata = []
  expected_parts = []
  for segment_index in expected_indices:
    metadata, tokens = _assemble(("expected", segment_index))
    expected_metadata.append(metadata)
    expected_parts.append(tokens)
  if (
      actual_metadata.get("kind") != "serving_prompt"
      or actual_metadata.get("turn_index") != header["turn"]
      or "done" in actual_metadata
      or expected_metadata[0].get("kind") != "initial_prompt"
      or expected_metadata[0].get("turn_index") != -1
      or "done" in expected_metadata[0]
  ):
    raise ValueError("first-diff stream attribution is inconsistent")
  cursor = 1
  for step_index in range(header["trajectory_steps"]):
    if cursor >= len(expected_metadata):
      raise ValueError("first-diff expected trajectory is incomplete")
    assistant = expected_metadata[cursor]
    if (
        assistant.get("kind") != "assistant"
        or assistant.get("turn_index") != step_index
        or type(assistant.get("done")) is not bool
    ):
      raise ValueError("first-diff assistant attribution is inconsistent")
    cursor += 1
    has_environment = (
        cursor < len(expected_metadata)
        and expected_metadata[cursor].get("kind") == "environment"
        and expected_metadata[cursor].get("turn_index") == step_index
    )
    if has_environment:
      environment = expected_metadata[cursor]
      if (
          type(environment.get("done")) is not bool
          or environment["done"] != assistant["done"]
      ):
        raise ValueError("first-diff environment attribution is inconsistent")
      cursor += 1
    elif not assistant["done"]:
      raise ValueError("first-diff nonterminal environment segment is missing")
    if assistant["done"] and step_index != header["trajectory_steps"] - 1:
      raise ValueError("first-diff terminal turn is not final")
  if cursor != len(expected_metadata):
    raise ValueError("first-diff expected trajectory has extra segments")
  expected = np.concatenate(expected_parts, axis=0)
  if (
      actual.size != header.get("actual_tokens")
      or expected.size != header.get("expected_tokens")
      or _digest(actual) != header.get("actual_sha256")
      or _digest(expected) != header.get("expected_sha256")
  ):
    raise ValueError("first-diff whole-stream hash/length differs")
  common = min(actual.size, expected.size)
  unequal = np.flatnonzero(actual[:common] != expected[:common])
  first_mismatch = (
      int(unequal[0])
      if unequal.size
      else common
      if actual.size != expected.size
      else -1
  )
  if first_mismatch < 0 or first_mismatch != header.get("first_mismatch"):
    raise ValueError("first-diff mismatch coordinate is absent or inconsistent")
  return {
      "schema": "p57-token-first-diff-capsule-v1",
      "header": header,
      "actual": actual_metadata,
      "expected_segments": expected_metadata,
  }


def write_continuity_debug_capsule(
    lines: Sequence[str],
    *,
    state_dir: str | os.PathLike[str] | None = None,
) -> tuple[Path, str, int]:
  """Atomically persists one verified first-diff replay capsule."""
  capsule = debug_capsule_from_receipts(lines)
  root_value = state_dir if state_dir is not None else os.environ.get("CANON_STATE")
  if not root_value:
    raise ValueError("CANON_STATE is required for first-diff capsule persistence")
  root = Path(root_value)
  if not root.is_absolute():
    raise ValueError("first-diff capsule state directory must be absolute")
  output_dir = root / "token-continuity-first-diff"
  output_dir.mkdir(parents=True, exist_ok=True)
  output_dir.chmod(0o700)
  header = capsule["header"]
  filename = (
      f"p57-{header['workload']}-turn{header['turn']}-"
      f"{header['capsule_id']}-{os.getpid()}-{time.time_ns()}.json"
  )
  target = output_dir / filename
  partial = output_dir / f".{filename}.partial"
  payload = (
      json.dumps(capsule, sort_keys=True, separators=(",", ":")) + "\n"
  ).encode("utf-8")
  descriptor = os.open(
      partial,
      os.O_WRONLY | os.O_CREAT | os.O_EXCL,
      0o600,
  )
  try:
    with os.fdopen(descriptor, "wb") as output:
      output.write(payload)
      output.flush()
      os.fsync(output.fileno())
    os.rename(partial, target)
  except BaseException:
    try:
      partial.unlink()
    except FileNotFoundError:
      pass
    raise
  return target, hashlib.sha256(payload).hexdigest(), len(payload)


def prompt_token_witness_record(
    witness: Any,
    *,
    workload: str,
    trajectory_id: str,
    turn: int,
    pair_index: Any = None,
    group_id: Any = None,
) -> dict[str, Any]:
  """Builds one token-content-free submitted/RequestOutput witness record."""
  if workload not in ("p45", "m15"):
    raise ValueError(f"unsupported FrozenLake witness workload: {workload!r}")
  if (
      not isinstance(trajectory_id, str)
      or len(trajectory_id) != 32
      or any(character not in "0123456789abcdef" for character in trajectory_id)
  ):
    raise ValueError("prompt witness trajectory_id must be 32 lowercase hex")
  if not isinstance(turn, (int, np.integer)) or int(turn) < 0:
    raise ValueError("prompt witness turn must be a nonnegative integer")
  request_id = getattr(witness, "request_id", None)
  if not isinstance(request_id, str) or not request_id:
    raise ValueError("prompt witness request ID is absent")
  submitted_tokens = getattr(witness, "submitted_tokens", None)
  engine_echo_tokens = getattr(witness, "engine_echo_tokens", None)
  if (
      not isinstance(submitted_tokens, (int, np.integer))
      or not isinstance(engine_echo_tokens, (int, np.integer))
      or int(submitted_tokens) <= 0
      or int(engine_echo_tokens) <= 0
  ):
    raise ValueError("prompt witness token counts must be positive integers")
  submitted_sha = getattr(witness, "submitted_sha256", None)
  engine_echo_sha = getattr(witness, "engine_echo_sha256", None)
  for name, value in (
      ("submitted", submitted_sha),
      ("engine_echo", engine_echo_sha),
  ):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
      raise ValueError(f"prompt witness {name} SHA256 is invalid")
  return {
      "schema": "canon.p57-tito-host-witness.v1",
      "request_id": request_id,
      "trajectory_id": trajectory_id,
      "workload": workload,
      "turn": int(turn),
      "pair_index": None if pair_index is None else str(pair_index),
      "group_id": None if group_id is None else str(group_id),
      "submitted_tokens": int(submitted_tokens),
      "submitted_sha256": submitted_sha,
      "engine_echo_tokens": int(engine_echo_tokens),
      "engine_echo_sha256": engine_echo_sha,
      "submitted_equals_engine_echo": bool(
          int(submitted_tokens) == int(engine_echo_tokens)
          and submitted_sha == engine_echo_sha
      ),
  }


def write_prompt_token_witness(
    record: Mapping[str, Any],
    *,
    state_dir: str | os.PathLike[str] | None = None,
) -> tuple[Path, str, int]:
  """Atomically persists one submitted/engine-echo witness record."""
  if record.get("schema") != "canon.p57-tito-host-witness.v1":
    raise ValueError("invalid P57 TiTO host witness schema")
  request_id = record.get("request_id")
  if not isinstance(request_id, str) or not request_id:
    raise ValueError("P57 TiTO host witness request ID is absent")
  root_value = state_dir if state_dir is not None else os.environ.get("CANON_STATE")
  if not root_value:
    raise ValueError("CANON_STATE is required for prompt witness persistence")
  root = Path(root_value)
  if not root.is_absolute():
    raise ValueError("prompt witness state directory must be absolute")
  output_dir = root / "p57_tito_witness" / "host"
  output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
  output_dir.parent.chmod(0o700)
  output_dir.chmod(0o700)
  request_key = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16]
  filename = f"host-request-{request_key}.json"
  target = output_dir / filename
  partial = output_dir / f".{filename}.partial-{os.getpid()}-{time.time_ns()}"
  payload = (
      json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n"
  ).encode("utf-8")
  descriptor = os.open(
      partial,
      os.O_WRONLY | os.O_CREAT | os.O_EXCL,
      0o600,
  )
  try:
    with os.fdopen(descriptor, "wb") as output:
      output.write(payload)
      output.flush()
      os.fsync(output.fileno())
    try:
      os.link(partial, target)
    except FileExistsError as error:
      raise FileExistsError(
          f"duplicate P57 TiTO host witness: {request_id}"
      ) from error
    partial.unlink()
  except BaseException:
    try:
      partial.unlink()
    except FileNotFoundError:
      pass
    raise
  return target, hashlib.sha256(payload).hexdigest(), len(payload)


def write_prompt_echo_difference_capsule(
    witness: Any,
    record: Mapping[str, Any],
    *,
    state_dir: str | os.PathLike[str] | None = None,
    event_index: int | None = None,
) -> tuple[Path, str, int]:
  """Persists the raw submitted/echo IDs for one attributable echo red."""
  submitted = _integer_vector(
      getattr(witness, "submitted_token_ids", None),
      field="P57 submitted prompt echo capsule",
  )
  echoed = _integer_vector(
      getattr(witness, "engine_echo_token_ids", None),
      field="P57 engine prompt echo capsule",
  )
  if token_streams_equal(submitted, echoed):
    raise ValueError("prompt echo difference capsule requires unequal streams")
  if event_index is not None and (
      type(event_index) is not int or event_index <= 0
  ):
    raise ValueError("prompt echo capsule event_index must be positive")
  if (
      record.get("submitted_sha256") != _prompt_witness_digest(submitted)
      or record.get("engine_echo_sha256") != _prompt_witness_digest(echoed)
  ):
    raise ValueError("prompt echo capsule differs from its length/SHA witness")
  root_value = state_dir if state_dir is not None else os.environ.get("CANON_STATE")
  if not root_value or not os.path.isabs(os.fspath(root_value)):
    raise ValueError("prompt echo capsule requires an absolute CANON_STATE")
  output_dir = Path(root_value) / "token-continuity-first-diff"
  output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
  output_dir.chmod(0o700)
  request_key = hashlib.sha256(
      str(record["request_id"]).encode("utf-8")
  ).hexdigest()[:16]
  target = output_dir / (
      f"p57-{record['workload']}-echo-{request_key}-{os.getpid()}-"
      f"{time.time_ns()}.json"
  )
  capsule = {
      "schema": "canon.p57-tito-echo-diff.v1",
      "witness": dict(record),
      "submitted_token_ids": [int(token) for token in submitted],
      "engine_echo_token_ids": [int(token) for token in echoed],
  }
  if event_index is not None:
    capsule["event_index"] = event_index
  payload = (json.dumps(capsule, sort_keys=True, separators=(",", ":")) + "\n").encode()
  partial = output_dir / f".{target.name}.partial"
  descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
  try:
    with os.fdopen(descriptor, "wb") as output:
      output.write(payload)
      output.flush()
      os.fsync(output.fileno())
    os.link(partial, target)
    partial.unlink()
  except BaseException:
    partial.unlink(missing_ok=True)
    raise
  return target, hashlib.sha256(payload).hexdigest(), len(payload)


def write_tito_diagnostic_summary(
    record: Mapping[str, Any],
    *,
    state_dir: str | os.PathLike[str] | None = None,
) -> tuple[Path, str, int]:
  """Atomically persists the bounded rollout-only diagnostic summary."""
  if record.get("schema") != "canon.p57-tito-diagnostic.v1":
    raise ValueError("invalid P57 TiTO diagnostic summary schema")
  root_value = state_dir if state_dir is not None else os.environ.get("CANON_STATE")
  if not root_value:
    raise ValueError("CANON_STATE is required for TiTO diagnostic summary")
  root = Path(root_value)
  if not root.is_absolute():
    raise ValueError("TiTO diagnostic state directory must be absolute")
  output_dir = root / "p57_tito_witness"
  output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
  output_dir.chmod(0o700)
  target = output_dir / "diagnostic-summary.json"
  partial = output_dir / (
      f".diagnostic-summary.json.partial-{os.getpid()}-{time.time_ns()}"
  )
  payload = (
      json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n"
  ).encode("utf-8")
  descriptor = os.open(
      partial,
      os.O_WRONLY | os.O_CREAT | os.O_EXCL,
      0o600,
  )
  try:
    with os.fdopen(descriptor, "wb") as output:
      output.write(payload)
      output.flush()
      os.fsync(output.fileno())
    try:
      os.link(partial, target)
    except FileExistsError as error:
      raise FileExistsError(
          "refusing to overwrite P57 TiTO diagnostic summary"
      ) from error
    partial.unlink()
  except BaseException:
    try:
      partial.unlink()
    except FileNotFoundError:
      pass
    raise
  return target, hashlib.sha256(payload).hexdigest(), len(payload)


def append_full_record_batch_map(
    rows: Sequence[Mapping[str, Any]],
    *,
    state_dir: str | os.PathLike[str] | None = None,
) -> Path:
  """Appends one contiguous group within a step's trajectory-to-row join."""
  if not rows:
    raise ValueError("record-full row map cannot be empty")
  root_value = state_dir if state_dir is not None else os.environ.get("CANON_STATE")
  if not root_value or not os.path.isabs(os.fspath(root_value)):
    raise ValueError("record-full row map requires an absolute CANON_STATE")
  validated = []
  seen_rows = set()
  seen_ids = set()
  seen_request_ids = set()
  steps = set()
  for raw in rows:
    record = dict(raw)
    trajectory_id = record.get("trajectory_id")
    row = record.get("sequence_row")
    step = record.get("policy_step")
    request_ids = record.get("request_ids")
    if (
        not isinstance(trajectory_id, str)
        or len(trajectory_id) != 32
        or any(character not in "0123456789abcdef" for character in trajectory_id)
        or type(row) is not int
        or row < 0
        or type(step) is not int
        or step < 0
        or not isinstance(request_ids, list)
        or not request_ids
        or any(
            not isinstance(request_id, str) or not request_id
            for request_id in request_ids
        )
        or len(set(request_ids)) != len(request_ids)
    ):
      raise ValueError("record-full row identity is malformed")
    if (
        row in seen_rows
        or trajectory_id in seen_ids
        or seen_request_ids.intersection(request_ids)
    ):
      raise ValueError("record-full row map contains duplicate identity")
    seen_rows.add(row)
    seen_ids.add(trajectory_id)
    seen_request_ids.update(request_ids)
    steps.add(step)
    validated.append({"schema": "canon.p57-tito-row-map.v1", **record})
  first_row = min(seen_rows)
  if (
      len(steps) != 1
      or seen_rows != set(range(first_row, first_row + len(validated)))
  ):
    raise ValueError("record-full row map is not one contiguous policy group")
  output = Path(root_value) / "p57_tito_witness" / "full-row-map.jsonl"
  output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
  output.parent.chmod(0o700)
  descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
  with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
    for record in validated:
      stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    stream.flush()
    os.fsync(stream.fileno())
  return output


def consume_actor_snapshot_request(
    actor_trainer: Any,
    *,
    step: int,
    state_dir: str | os.PathLike[str] | None = None,
    manager_factory: Any = None,
    state_inspector: Any = None,
) -> dict[str, Any] | None:
  """Saves one requested pre-update actor-only replay snapshot.

  Request identity defects are fatal. Storage failure is recorded and returned
  so the unchanged training row can continue; terminal evidence classification
  will then fail.
  """
  if frozenlake_token_continuity_debug_mode(os.environ) != (
      P57_TOKEN_CONTINUITY_DEBUG_RECORD_FULL
  ):
    return None
  if os.environ.get(P57_TITO_ONEHOST_NEUTRALITY_ENV) == "on":
    return None
  root_value = state_dir if state_dir is not None else os.environ.get("CANON_STATE")
  if not root_value or not os.path.isabs(os.fspath(root_value)):
    raise ValueError("actor snapshot consumer requires an absolute CANON_STATE")
  root = Path(root_value)
  request_dir = root / "p57_tito_witness" / "actor-snapshot-requests"
  receipt_dir = root / "p57_tito_witness" / "actor-snapshot-receipts"
  source_commit = os.environ.get("CANON_EXPECT_COMMIT", "")
  image_identity = os.environ.get("CANON_CLIENT_IMAGE", "")
  workload_key = (
      os.environ.get("CANON_P57_WORKLOAD_CANDIDATE", ""),
      os.environ.get("CANON_P57_DATA_SPLIT", ""),
  )
  expected_workload = (
      "p45" if workload_key == ("", "")
      else "m15" if workload_key == ("m15", "main")
      else ""
  )
  if (
      re.fullmatch(r"[0-9a-f]{40}", source_commit) is None
      or not image_identity
      or not expected_workload
      or os.environ.get("CANON_DP_SIZE", "") != "8"
      or os.environ.get("CANON_TP_SIZE", "") != "8"
  ):
    raise ValueError(
        "actor snapshot consumer escaped the P45/M15 DP8xTP8 source/image "
        "identity"
    )
  requests = sorted(request_dir.glob("step-*.json"))
  pending_by_step = {}
  seen_categories = set()
  for path in requests:
    match = re.fullmatch(r"step-([0-9]{6})\.json", path.name)
    try:
      request = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
      raise ValueError(f"actor snapshot request is unreadable: {path.name}") from exc
    categories = request.get("categories")
    request_step = request.get("step")
    if (
        match is None
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_mode & 0o077
        or request.get("schema")
        != "canon.p57-tito-actor-snapshot-request.v1"
        or request.get("status") != "PENDING"
        or type(request_step) is not int
        or request_step != int(match.group(1))
        or request.get("policy_version") != request_step
        or not isinstance(categories, list)
        or not categories
        or len(set(categories)) != len(categories)
        or any(category not in P57_ACTOR_SNAPSHOT_THRESHOLDS for category in categories)
        or seen_categories.intersection(categories)
        or not isinstance(request.get("max_abs"), (int, float))
        or not np.isfinite(request["max_abs"])
        or any(
            float(request["max_abs"])
            < P57_ACTOR_SNAPSHOT_THRESHOLDS[category]
            for category in categories
        )
        or not isinstance(request.get("sidecar_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", request["sidecar_sha256"]) is None
        or request.get("source_commit") != source_commit
        or request.get("image_identity") != image_identity
        or request.get("workload") != expected_workload
        or request.get("dp") != 8
        or request.get("tp") != 8
    ):
      raise ValueError(f"actor snapshot request identity differs: {path.name}")
    seen_categories.update(categories)
    pending_by_step[request_step] = (path, request)
  if (
      len(pending_by_step) > len(P57_ACTOR_SNAPSHOT_THRESHOLDS)
      or len(seen_categories) > len(P57_ACTOR_SNAPSHOT_THRESHOLDS)
  ):
    raise ValueError("actor snapshot requests exceed the registered bound")
  for request_step in pending_by_step:
    if request_step < step and not (
        receipt_dir / f"step-{request_step:06d}.json"
    ).is_file():
      raise ValueError(
          f"actor snapshot request became stale before step {step}: {request_step}"
      )
  if step not in pending_by_step:
    return None
  if int(getattr(actor_trainer, "train_steps")) != step:
    raise ValueError(
        "actor snapshot request differs from trainer pre-update step: "
        f"request={step} trainer={getattr(actor_trainer, 'train_steps', None)}"
    )
  request_path, request = pending_by_step[step]
  receipt_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
  receipt_dir.chmod(0o700)
  receipt_path = receipt_dir / f"step-{step:06d}.json"
  if receipt_path.exists():
    raise FileExistsError(
        f"actor snapshot request was consumed twice: {receipt_path.name}"
    )
  prefix = os.environ.get("CANON_P57_TITO_GCS_PREFIX", "").rstrip("/")
  if _P57_TITO_GCS_PREFIX_RE.fullmatch(prefix) is None:
    raise ValueError("actor snapshot root is outside the registered evidence root")
  snapshot_root = f"{prefix}/actor-snapshots"
  if manager_factory is None:
    from tunix.sft import checkpoint_manager as checkpoint_manager_lib  # pylint: disable=g-import-not-at-top
    from tunix.sft import checkpoint_options as checkpoint_options_lib  # pylint: disable=g-import-not-at-top

    options = checkpoint_options_lib.TunixCheckpointingOptions(
        enable_async_checkpointing=False,
        save_on_close=False,
    )

    def manager_factory(value):
      return checkpoint_manager_lib.CheckpointManager(
          root_directory=value, options=options
      )

  if state_inspector is None:
    from flax import nnx  # pylint: disable=g-import-not-at-top
    import jax  # pylint: disable=g-import-not-at-top

    def state_inspector(trainer):
      params = nnx.state(trainer.model)
      flattened = jax.tree_util.tree_flatten_with_path(
          params, is_leaf=lambda value: isinstance(value, nnx.Variable)
      )[0]
      leaves = []
      logical_bytes = 0
      for path, value in flattened:
        array = value[...] if isinstance(value, nnx.Variable) else value
        shape = tuple(getattr(array, "shape", ()))
        dtype = getattr(array, "dtype", None)
        if dtype is None:
          raise ValueError("actor snapshot model leaf has no dtype")
        itemsize = int(np.dtype(dtype).itemsize)
        size = int(np.prod(shape, dtype=np.int64)) if shape else 1
        logical_bytes += size * itemsize
        leaves.append({
            "path": jax.tree_util.keystr(path),
            "shape": list(shape),
            "dtype": str(dtype),
            "logical_bytes": size * itemsize,
        })
      if not leaves:
        raise ValueError("actor snapshot model has no leaves")
      return {
          "leaves": leaves,
          "leaf_count": len(leaves),
          "logical_bytes": logical_bytes,
          "bounded_fingerprint": trainer._canon_fingerprint_state(params),  # pylint: disable=protected-access
      }

  started = time.perf_counter()
  status = "FAIL"
  latest_step = None
  inspection = None
  failure_type = None
  manager = None
  try:
    inspection = state_inspector(actor_trainer)
    manager = manager_factory(snapshot_root)
    saved = manager.save(
        step,
        actor_trainer.model,
        optimizer=None,
        force=True,
        custom_metadata={
            "schema": "canon.p57-tito-actor-snapshot.v1",
            "artifact_kind": "actor-only-nonresumable",
            "source_commit": request.get("source_commit"),
            "image_identity": request.get("image_identity"),
            "workload": request.get("workload"),
            "dp": request.get("dp"),
            "tp": request.get("tp"),
            "policy_version": step,
            "categories": request["categories"],
            "max_abs": request["max_abs"],
            "optimizer_included": False,
            "resumable": False,
            "bounded_fingerprint": inspection["bounded_fingerprint"],
        },
    )
    latest_step = manager.latest_step()
    if saved is not True or latest_step != step:
      raise RuntimeError("actor snapshot save did not close on its requested step")
    status = "PASS"
  except Exception as exc:  # pylint: disable=broad-exception-caught
    failure_type = type(exc).__name__
  finally:
    if manager is not None:
      try:
        manager.close()
      except Exception as exc:  # pylint: disable=broad-exception-caught
        status = "FAIL"
        failure_type = type(exc).__name__
  receipt = {
      "schema": "canon.p57-tito-actor-snapshot-receipt.v1",
      "status": status,
      "step": step,
      "policy_version": step,
      "categories": request["categories"],
      "max_abs": request["max_abs"],
      "source_commit": request["source_commit"],
      "image_identity": request["image_identity"],
      "workload": request["workload"],
      "dp": request["dp"],
      "tp": request["tp"],
      "request_path": str(request_path),
      "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
      "snapshot_root": snapshot_root,
      "snapshot_root_sha256": hashlib.sha256(snapshot_root.encode()).hexdigest(),
      "latest_step": latest_step,
      "optimizer_included": False,
      "resumable": False,
      "actor_train_steps_before": int(actor_trainer.train_steps),
      "actor_train_steps_after": int(actor_trainer.train_steps),
      "save_seconds": time.perf_counter() - started,
      "model_inventory": inspection,
      "failure_type": failure_type,
  }
  payload = (
      json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
  ).encode("utf-8")
  partial = receipt_path.with_name(
      f".{receipt_path.name}.partial-{os.getpid()}-{time.time_ns()}"
  )
  descriptor = os.open(
      partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
  )
  try:
    with os.fdopen(descriptor, "wb") as output:
      output.write(payload)
      output.flush()
      os.fsync(output.fileno())
    os.link(partial, receipt_path)
    partial.unlink()
  except BaseException:
    try:
      os.close(descriptor)
    except OSError:
      pass
    partial.unlink(missing_ok=True)
    raise
  print(
      "[P57.TITO.ACTOR_SNAPSHOT] "
      f"status={status} step={step} categories={','.join(request['categories'])} "
      f"optimizer_included=0 resumable=0 latest_step={latest_step} "
      f"save_seconds={receipt['save_seconds']:.6f} "
      f"receipt_sha256={hashlib.sha256(payload).hexdigest()}",
      flush=True,
  )
  return receipt


def write_tito_full_record_summary(
    record: Mapping[str, Any],
    *,
    state_dir: str | os.PathLike[str] | None = None,
) -> tuple[Path, str, int]:
  """Atomically persists measured full-training TiTO accounting."""
  if record.get("schema") != "canon.p57-tito-full-record.v1":
    raise ValueError("invalid P57 TiTO full-record summary schema")
  root_value = state_dir if state_dir is not None else os.environ.get("CANON_STATE")
  if not root_value or not os.path.isabs(os.fspath(root_value)):
    raise ValueError("TiTO full-record summary requires an absolute CANON_STATE")
  output_dir = Path(root_value) / "p57_tito_witness"
  output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
  output_dir.chmod(0o700)
  target = output_dir / "full-record-summary.json"
  payload = (json.dumps(dict(record), sort_keys=True, separators=(",", ":")) + "\n").encode()
  partial = output_dir / f".{target.name}.partial-{os.getpid()}-{time.time_ns()}"
  descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
  try:
    with os.fdopen(descriptor, "wb") as output:
      output.write(payload)
      output.flush()
      os.fsync(output.fileno())
    os.link(partial, target)
    partial.unlink()
  except BaseException:
    partial.unlink(missing_ok=True)
    raise
  return target, hashlib.sha256(payload).hexdigest(), len(payload)


def continuity_receipt(
    actual: Sequence[int] | np.ndarray,
    expected: Sequence[int] | np.ndarray,
    *,
    turn: int,
    mode: str = "verify",
    workload: str = "m15",
    selector: str = M15_TOKEN_CONTINUITY_ENV,
    trajectory_id: str | None = None,
) -> str:
  """Builds a bounded, token-content-free equality receipt."""
  if mode not in ("verify", "exact"):
    raise ValueError(f"unsupported FrozenLake token-continuity mode: {mode!r}")
  if workload not in ("p45", "m15"):
    raise ValueError(f"unsupported FrozenLake token-continuity workload: {workload!r}")
  if selector not in (M15_TOKEN_CONTINUITY_ENV, P57_TOKEN_CONTINUITY_ENV):
    raise ValueError(f"unsupported token-continuity selector: {selector!r}")
  if selector == M15_TOKEN_CONTINUITY_ENV and workload != "m15":
    raise ValueError("the historical M15 selector cannot attest P45")
  if trajectory_id is not None and (
      len(trajectory_id) != 32
      or any(character not in "0123456789abcdef" for character in trajectory_id)
  ):
    raise ValueError("token-continuity trajectory_id must be 32 lowercase hex")
  actual_tokens = _integer_vector(
      actual, field=f"{workload.upper()} actual prompt tokens"
  )
  expected_tokens = _integer_vector(
      expected, field=f"{workload.upper()} expected prompt tokens"
  )
  common = min(actual_tokens.size, expected_tokens.size)
  unequal = np.flatnonzero(actual_tokens[:common] != expected_tokens[:common])
  if unequal.size:
    first_mismatch = int(unequal[0])
    actual_token = str(int(actual_tokens[first_mismatch]))
    expected_token = str(int(expected_tokens[first_mismatch]))
  elif actual_tokens.size != expected_tokens.size:
    first_mismatch = common
    actual_token = (
        str(int(actual_tokens[common]))
        if common < actual_tokens.size
        else "NA"
    )
    expected_token = (
        str(int(expected_tokens[common]))
        if common < expected_tokens.size
        else "NA"
    )
  else:
    first_mismatch = -1
    actual_token = "NA"
    expected_token = "NA"
  equal = first_mismatch == -1
  verdict = "TOKEN_STREAM_EQUAL" if equal else "TOKEN_STREAM_DIFFERENT"
  marker = (
      "[CANON_M15_TOKEN_CONTINUITY]"
      if selector == M15_TOKEN_CONTINUITY_ENV
      else "[CANON_P57_TOKEN_CONTINUITY]"
  )
  workload_field = (
      ""
      if selector == M15_TOKEN_CONTINUITY_ENV
      else f" workload={workload}"
  )
  trajectory_field = (
      "" if trajectory_id is None else f" trajectory_id={trajectory_id}"
  )
  return (
      f"{marker}{workload_field} mode={mode}{trajectory_field} "
      f"turn={turn} verdict={verdict} "
      f"actual_tokens={actual_tokens.size} "
      f"expected_tokens={expected_tokens.size} "
      f"actual_sha256={_digest(actual_tokens)} "
      f"expected_sha256={_digest(expected_tokens)} "
      f"first_mismatch={first_mismatch} actual_token={actual_token} "
      f"expected_token={expected_token}"
  )


def token_streams_equal(
    actual: Sequence[int] | np.ndarray,
    expected: Sequence[int] | np.ndarray,
) -> bool:
  """Returns exact token-stream equality after validating both operands."""
  actual_tokens = _integer_vector(
      actual, field="FrozenLake actual prompt tokens"
  )
  expected_tokens = _integer_vector(
      expected, field="FrozenLake expected prompt tokens"
  )
  return bool(np.array_equal(actual_tokens, expected_tokens))
