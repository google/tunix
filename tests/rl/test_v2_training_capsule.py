"""Exact-input capsule gates for the V2 FrozenLake one-host carrier."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import types

import numpy as np
import pytest

from tunix.rl import p64_training_capsule as capsule


def _env(
    path: Path, *, mode: str, workload: str = "p45"
) -> dict[str, str]:
  workload_name = {
      "p45": "frozenlake-p45-onehost-dp2-tp2",
      "m15": "frozenlake-m15-onehost-dp2-tp2",
  }[workload]
  return {
      "CANON_V2_TRAINING_CAPSULE_MODE": mode,
      "CANON_V2_TRAINING_CAPSULE": str(path),
      "CANON_V2_TRAINING_CAPSULE_SHA256": "",
      "CANON_V2_MODEL_BINDING_SHA256": "",
      "CANON_PROFILE_FILE": (
          "cluster/profiles/qwen3-8b-dp2-tp2-frozenlake-onehost.env"
      ),
      "CANON_P32_WORKLOAD": workload_name,
      "CANON_DP_SIZE": "2",
      "CANON_TP_SIZE": "2",
      "CANON_GLOBAL_TRAJECTORIES": "16",
      "CANON_LOCAL_TRAJECTORIES": "8",
      "CANON_LOGPROB_M": "256",
      "FL_SHARED_MESH": "2,2",
      "CANON_P33_RUN_STAGE": "backward-no-commit",
      "CANON_P33_NO_COMMIT": "1",
      "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
      "CANON_P66_P59_CHECK_VMA": "1",
      "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
      "CANON_V1_HP_FULL": "0",
      "CANON_MODEL_DIR_NAME": "qwen8b_tp2",
      "CANON_P64_P45_NUMERIC_DEBUG": "0",
      "V2_FL_WORKLOAD": workload,
      "V2_FL_MODE": "measure" if mode == "capture" else "certify",
      "V2_FL_SOURCE_SHA": "a" * 40,
      "V2_FL_LABEL": f"v2-capsule-{mode}",
  }


def _observed(rows: int = 16, completion_width: int = 2048):
  prompt_ids = np.arange(rows * 4096, dtype=np.int32).reshape(rows, 4096)
  prompt_mask = np.ones_like(prompt_ids, dtype=np.bool_)
  completion_ids = np.arange(
      rows * completion_width, dtype=np.int32
  ).reshape(rows, completion_width)
  completion_mask = np.ones_like(completion_ids, dtype=np.bool_)
  logps = np.arange(
      rows * completion_width, dtype=np.float32
  ).reshape(rows, completion_width)
  policy_version = np.arange(rows, dtype=np.int32)
  train = types.SimpleNamespace(
      prompt_ids=prompt_ids,
      prompt_mask=prompt_mask,
      completion_ids=completion_ids,
      completion_mask=completion_mask,
      advantages=np.arange(rows, dtype=np.float32),
      ref_per_token_logps=logps + 1,
      old_per_token_logps=logps,
      segment_ids=np.zeros_like(completion_ids, dtype=np.int32),
      segment_positions=np.broadcast_to(
          np.arange(completion_width, dtype=np.int32), completion_ids.shape
      ),
      is_update_step=np.asarray(True),
      sampler_is_weights=np.ones_like(logps),
      policy_version=policy_version,
      completion_valid_mask=completion_mask,
  )
  return types.SimpleNamespace(
      train_example=train,
      s_decode=logps.copy(),
      s_prefill=logps.copy(),
      t_old=logps.copy(),
      action_mask=completion_mask,
      completion_valid_mask=completion_mask,
      prompt_mask=prompt_mask,
      tokens=completion_ids,
      policy_version=policy_version,
      sampling_values=np.zeros((rows, 3), dtype=np.float32),
      source_name="v2-unit-rescore",
      all_compact_filtered=False,
  )


def test_v2_capsule_round_trip_is_array_and_model_hash_bound(capsys):
  with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / "v2_training_capsule.npz"
    capture_env = _env(path, mode="capture")
    observed = _observed()
    capture = capsule.persist(observed, capture_env)
    fingerprint = {"leaves": {"x": {"sha256": "1" * 64}}}
    binding = capsule.bind_or_verify_model(fingerprint, capture_env)
    replay_env = {
        **_env(path, mode="replay"),
        "CANON_V2_TRAINING_CAPSULE_SHA256": capture["sha256"],
        "CANON_V2_MODEL_BINDING_SHA256": binding["binding_sha256"],
        "V2_FL_SOURCE_SHA": "b" * 40,
    }
    verified = capsule.load_verified(replay_env)
    rebuilt = verified.build(types.SimpleNamespace, types.SimpleNamespace)
    for field in (
        "prompt_ids",
        "completion_ids",
        "advantages",
        "ref_per_token_logps",
        "old_per_token_logps",
        "segment_ids",
        "segment_positions",
        "sampler_is_weights",
    ):
      assert np.array_equal(
          getattr(rebuilt.train_example, field),
          getattr(observed.train_example, field),
      )
    capsule.bind_or_verify_model(fingerprint, replay_env)
    assert capsule.reverse_group_limit(8, replay_env) == 8
    assert capsule.reverse_group_limit(8, capture_env) == 8
    receipts = capsys.readouterr().out
    assert "[V2.CAPSULE] capture_ready" in receipts
    assert " rows=16 " in receipts

    wrong_hash = copy.deepcopy(replay_env)
    wrong_hash["CANON_V2_TRAINING_CAPSULE_SHA256"] = "f" * 64
    with pytest.raises(capsule.P64TrainingCapsuleError, match="file hash"):
      capsule.load_verified(wrong_hash)
    with pytest.raises(capsule.P64TrainingCapsuleError, match="live-model"):
      capsule.bind_or_verify_model({"leaves": {}}, replay_env)
    with pytest.raises(capsule.P64TrainingCapsuleError, match="8 registered"):
      capsule.reverse_group_limit(7, replay_env)


def test_v2_capsule_identity_and_namespace_fail_closed():
  with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / "v2_training_capsule.npz"
    values = _env(path, mode="capture")
    capsule.validate_identity(values)

    wrong_tp = {**values, "CANON_TP_SIZE": "1"}
    with pytest.raises(capsule.P64TrainingCapsuleError, match="identity drifted"):
      capsule.validate_identity(wrong_tp)

    replay_profile = _env(path, mode="replay")
    replay_profile["V2_FL_MODE"] = "profile"
    capsule.validate_identity(replay_profile)
    for capsule_mode, vehicle_mode in (
        ("capture", "profile"),
        ("capture", "certify"),
        ("replay", "measure"),
        ("replay", "invalid"),
    ):
      wrong_vehicle_mode = _env(path, mode=capsule_mode)
      wrong_vehicle_mode["V2_FL_MODE"] = vehicle_mode
      with pytest.raises(
          capsule.P64TrainingCapsuleError, match="identity drifted"
      ):
        capsule.validate_identity(wrong_vehicle_mode)

    mixed = {
        **values,
        "CANON_P64_TRAINING_CAPSULE_MODE": "capture",
    }
    with pytest.raises(capsule.P64TrainingCapsuleError, match="mutually exclusive"):
      capsule.mode(mixed)

    dangling = {
        "CANON_V2_TRAINING_CAPSULE": str(path),
    }
    with pytest.raises(capsule.P64TrainingCapsuleError, match="fields require"):
      capsule.mode(dangling)


def test_v2_m15_capsule_round_trip_uses_full_hard_context_width():
  with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / "v2_m15_training_capsule.npz"
    capture_env = _env(path, mode="capture", workload="m15")
    result = capsule.persist(
        _observed(completion_width=8192), capture_env
    )
    replay_env = {
        **_env(path, mode="replay", workload="m15"),
        "CANON_V2_TRAINING_CAPSULE_SHA256": result["sha256"],
    }
    verified = capsule.load_verified(replay_env)
    assert verified.metadata["completion_width"] == 8192
    assert verified.arrays["train__completion_ids"].shape == (16, 8192)
    assert capsule.reverse_group_limit(8, replay_env) == 8


def test_v2_capsule_rejects_wrong_physical_shape():
  with tempfile.TemporaryDirectory() as temporary:
    path = Path(temporary) / "v2_training_capsule.npz"
    with pytest.raises(capsule.P64TrainingCapsuleError, match="physical tensor"):
      capsule.persist(_observed(rows=15), _env(path, mode="capture"))


def test_v2_capsule_is_wired_after_strict_precheck_and_before_producer_bypass():
  root = Path(__file__).resolve().parents[2]
  grpo = (root / "tunix/rl/agentic/agentic_grpo_learner.py").read_text()
  learner = (root / "tunix/rl/agentic/agentic_rl_learner.py").read_text()
  assert grpo.index("alignment.check_pre_backward(") < grpo.index(
      "p64_training_capsule.persist(combined_batch)"
  )
  assert learner.index("p64_training_capsule.load_verified()") < learner.index(
      "producer_bypass verdict=PASS"
  )
  train_body = learner[learner.index("  def train(") :]
  assert train_body.index(
      "v2_capsule = p64_training_capsule.v2_enabled()"
  ) < train_body.index('capsule_marker = "V2.CAPSULE"')
  assert "if p64_capsule_mode:\n      p64_training_capsule.bind_or_verify_model" in learner
