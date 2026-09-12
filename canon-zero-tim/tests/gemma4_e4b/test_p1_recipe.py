"""Host gates for the isolated Gemma 4 E4B-IT P1 admission contract."""

from __future__ import annotations

import hashlib
import base64
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

REPO = Path(__file__).resolve().parents[3]
ADMISSION_PATH = REPO / "tunix/rl/gemma4_e4b_admission.py"
SPEC = importlib.util.spec_from_file_location("gemma4_e4b_admission", ADMISSION_PATH)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError(f"cannot load {ADMISSION_PATH}")
admission = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(admission)


def _load_script(name: str):
  path = REPO / "canon-zero-tim/tasks/gemma4-e4b-frozenlake/scripts" / name
  spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
  if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {path}")
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class P1RecipeTest(unittest.TestCase):

  def test_selector_is_default_off_and_closed(self):
    for value in (None, "", "0"):
      env = {} if value is None else {admission.SELECTOR_ENV: value}
      self.assertIsNone(admission.active_workload(env))
    self.assertEqual(admission.active_workload({admission.SELECTOR_ENV: "p45"}), "p45")
    self.assertEqual(admission.active_workload({admission.SELECTOR_ENV: "m15"}), "m15")
    with self.assertRaisesRegex(ValueError, "must be"):
      admission.active_workload({admission.SELECTOR_ENV: "full"})

  def test_p1_rejects_training_overrides_and_canonical_kernels(self):
    admission.require_stock_invocation((), {})
    admission.require_stock_invocation((), {"CANON_ENGINE_MODULE_C": "0"})
    with self.assertRaisesRegex(ValueError, "CLI recipe overrides"):
      admission.require_stock_invocation(("--num_batches=120",), {})
    with self.assertRaisesRegex(ValueError, "canonical runtime flags"):
      admission.require_stock_invocation((), {"CANON_P59_RANK_PARALLEL_BACKWARD": "1"})
    with self.assertRaisesRegex(ValueError, "prefix caching off"):
      admission.require_stock_invocation((), {"CANON_VLLM_ENABLE_PREFIX_CACHING": "1"})

  def test_e4b_it_config_matches_architecture_but_not_artifact_identity(self):
    model_source = (REPO / "tunix/models/gemma4/model.py").read_text()
    registry_source = (REPO / "tunix/models/registry.py").read_text()
    self.assertIn("def gemma4_e4b_it(", model_source)
    self.assertIn("return cls.gemma4_e4b(sharding_config=sharding_config)", model_source)
    self.assertIn("model_name='gemma-4-e4b'", registry_source)
    self.assertIn("model_name='gemma-4-e4b-it'", registry_source)
    self.assertIn("model_id='google/gemma-4-E4B-it'", registry_source)
    self.assertIn("model_version='e4b_it'", registry_source)

  def test_runtime_contract_accepts_only_real_target_geometry(self):
    base = dict(
        workload="p45", device_count=4, local_device_count=4,
        process_count=1, platforms=("tpu",), rollout_mesh_shape=(1, 4),
        device_kinds=("TPU v5",), device_ids=(0, 1, 2, 3),
        device_process_indices=(0, 0, 0, 0),
        trainer_mesh_shape=(1, 4), jax_version=admission.EXPECTED_JAX_VERSION,
        checkpoint_root=None,
        prefix_caching=False,
    )
    receipt = admission.require_runtime_contract(**base)
    self.assertEqual(receipt["target_geometry"], "DP1xTP4")
    for override in (
        {"device_count": 1}, {"process_count": 2},
        {"platforms": ("cpu",)}, {"rollout_mesh_shape": (2, 2)},
        {"device_kinds": ("TPU v5p",)}, {"device_ids": (0, 0, 2, 3)},
        {"device_process_indices": (0, 0, 0, 1)},
        {"jax_version": "0.0.0"},
        {"checkpoint_root": "/tmp/ckpt"}, {"prefix_caching": True},
    ):
      kwargs = {**base, **override}
      with self.assertRaisesRegex(RuntimeError, "runtime contract failed"):
        admission.require_runtime_contract(**kwargs)

  def test_shape_ledger_preserves_p45_and_m15_hard_caps(self):
    for workload, expected_cap in (("p45", 6400), ("m15", 12288)):
      receipt = admission.shape_ledger(
          workload, semantic_prompts=1, generations_per_prompt=1,
          mini_batch_size=1, train_micro_batch_size=1,
          compute_logps_micro_batch_size=1,
          max_num_seqs=4, max_num_batched_tokens=4096,
          kv_cache_size=expected_cap + 256,
      )
      self.assertEqual(receipt["context_hard_cap"], expected_cap)
      self.assertEqual(receipt["shard_local_rows"], 1)
    with self.assertRaisesRegex(RuntimeError, "below m15 hard cap"):
      admission.shape_ledger(
          "m15", semantic_prompts=1, generations_per_prompt=1,
          mini_batch_size=1, train_micro_batch_size=1,
          compute_logps_micro_batch_size=1,
          max_num_seqs=4, max_num_batched_tokens=4096,
          kv_cache_size=8192,
      )
    with self.assertRaisesRegex(RuntimeError, "mini/train/logps=1/1/1"):
      admission.shape_ledger(
          "p45", semantic_prompts=1, generations_per_prompt=1,
          mini_batch_size=1, train_micro_batch_size=2,
          compute_logps_micro_batch_size=2,
          max_num_seqs=4, max_num_batched_tokens=4096,
          kv_cache_size=6656,
      )

  def test_snapshot_verifier_checks_bytes_and_hashes(self):
    payloads = {"model.safetensors": b"weights", "config.json": b"config"}
    expected = {
        name: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for name, payload in payloads.items()
    }
    with tempfile.TemporaryDirectory() as tmp:
      root = Path(tmp)
      for name, payload in payloads.items():
        (root / name).write_bytes(payload)
      with mock.patch.object(admission, "SNAPSHOT_FILES", expected):
        receipt = admission.verify_snapshot_identity(tmp)
        self.assertEqual(set(receipt["files"]), set(payloads))
        (root / "config.json").write_bytes(b"drift")
        with self.assertRaisesRegex(RuntimeError, "size drifted"):
          admission.verify_snapshot_identity(tmp)

  def test_loaded_tokenizer_semantics_are_frozen(self):
    class GemmaTokenizer:
      chat_template = "template"
      bos_token_id = 2
      eos_token_id = 1
      pad_token_id = 0
      unk_token_id = 3

      def __len__(self):
        return 262144

      def encode(self, value, add_special_tokens):
        self.assert_no_special = not add_special_tokens
        return [len(value), 7]

    tokenizer = GemmaTokenizer()
    probe = {
        value: tokenizer.encode(value, add_special_tokens=False)
        for value in admission._TOKEN_PROBE_TEXTS
    }
    probe_sha = hashlib.sha256(
        json.dumps(probe, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    template_sha = hashlib.sha256(b"template").hexdigest()
    with (
        mock.patch.object(admission, "TOKEN_PROBE_SHA256", probe_sha),
        mock.patch.object(admission, "CHAT_TEMPLATE_SHA256", template_sha),
    ):
      receipt = admission.verify_tokenizer_identity(tokenizer)
    self.assertEqual(receipt["token_probe_sha256"], probe_sha)
    tokenizer.pad_token_id = 9
    with self.assertRaisesRegex(RuntimeError, "tokenizer identity drifted"):
      admission.verify_tokenizer_identity(tokenizer)

  def test_live_weight_gate_mirrors_gemma_preprocessing(self):
    class TextConfig:
      num_hidden_layers = 42
      num_attention_heads = 8
      hidden_size = 2560
      intermediate_size = 10240
      vocab_size = 262144
      sliding_window = 512
      final_logit_softcapping = 30.0
      vocab_size_per_layer_input = 262144
      hidden_size_per_layer_input = 256
      num_kv_shared_layers = 18
      tie_word_embeddings = True
      attention_k_eq_v = False
      enable_moe_block = False
      layer_types = [
          "full_attention" if (index + 1) % 6 == 0 else "sliding_attention"
          for index in range(42)
      ]

    class ModelConfig:
      hf_config = type("HFConfig", (), {"text_config": TextConfig()})()

      @staticmethod
      def get_total_num_kv_heads():
        return 2

      @staticmethod
      def get_head_size():
        return 256

    observed = []

    def preprocess(state):
      observed.append(("preprocess", state))
      return {"fused": state}

    def attest_fn(*, sampler, trainer_state, tp_size):
      observed.append(("attest", trainer_state, tp_size))
      return {
          "equal": True,
          "mismatch_indices": (),
          "mapped_leaves": 593,
          "live_leaves": 593,
          "mesh_shape": (("data", 1), ("model", 4)),
      }

    sampler = type("Sampler", (), {})()
    sampler.args = {"tensor_parallel_size": 4, "data_parallel_size": 1}
    sampler._model_runner = type("Runner", (), {"model_config": ModelConfig()})()
    sampler.config = type(
        "Config", (), {
            "mapping_config": type(
                "Mapping", (), {"preprocess_src_state": staticmethod(preprocess)}
            )()
        }
    )()
    with mock.patch.dict(os.environ, {admission.SELECTOR_ENV: "p45"}):
      receipt = admission.attest_exact_live_engine_weights(
          sampler=sampler, trainer_state="raw", attest_fn=attest_fn
      )
    self.assertTrue(receipt["equal"])
    self.assertEqual(receipt["architecture"]["num_kv_shared_layers"], 18)
    self.assertEqual(
        receipt["architecture"]["layer_types"], admission.EXPECTED_LAYER_TYPES
    )
    self.assertEqual(observed, [
        ("preprocess", "raw"),
        ("attest", {"fused": "raw"}, 4),
    ])
    TextConfig.num_kv_shared_layers = 17
    with (
        mock.patch.dict(os.environ, {admission.SELECTOR_ENV: "p45"}),
        self.assertRaisesRegex(RuntimeError, "live architecture drifted"),
    ):
      admission.attest_exact_live_engine_weights(
          sampler=sampler, trainer_state="raw", attest_fn=attest_fn
      )

  def test_profile_is_stock_bounded_and_default_off(self):
    profile = (
        REPO
        / "canon-zero-tim/cluster/profiles/"
        "gemma4-e4b-dp1-tp4-frozenlake.env"
    ).read_text(encoding="utf-8")
    self.assertIn("CANON_GEMMA4_E4B_P1_ADMISSION", profile)
    self.assertIn("FL_ROLLOUT_MESH=1,4", profile)
    self.assertIn("FL_TRAINER_MESH=1,4", profile)
    self.assertIn("CANON_ENGINE_MODULE_C=0", profile)
    self.assertIn("CANON_VLLM_ENABLE_PREFIX_CACHING=0", profile)
    self.assertNotIn("CANON_P59_RANK_PARALLEL_BACKWARD=1", profile)
    entrypoint = (
        REPO / "examples/frozenlake/train_frozenlake.py"
    ).read_text(encoding="utf-8")
    self.assertIn(
        "TRAIN_MICRO_BATCH_SIZE = 1 if P1_ADMISSION else 2", entrypoint
    )
    self.assertIn(
        "COMPUTE_LOGPS_MICRO_BATCH_SIZE = 1 if P1_ADMISSION else 2",
        entrypoint,
    )
    self.assertIn("train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE", entrypoint)
    self.assertIn(
        "compute_logps_micro_batch_size=COMPUTE_LOGPS_MICRO_BATCH_SIZE",
        entrypoint,
    )
    runner = (
        REPO
        / "canon-zero-tim/tasks/gemma4-e4b-frozenlake/scripts/"
        "run_p1_onehost_stock_admission.sh"
    ).read_text(encoding="utf-8")
    for name in (
        "CANON_PROFILE",
        "CANON_EXPECT_JAX_VERSION",
        "CANON_EXPECT_PATHWAYS_RELEASE",
    ):
      self.assertIn(f'-e {name}="${name}"', runner)
    self.assertIn(
        "sudo --preserve-env=HF_TOKEN docker run --rm --privileged", runner
    )
    self.assertIn("--env HF_TOKEN", runner)
    self.assertIn(
        'git -C "$repo" status --porcelain', runner
    )
    self.assertIn("refusing to launch P1 from a dirty runtime tree", runner)
    self.assertLess(
        runner.index('git -C "$repo" status --porcelain'),
        runner.index('mkdir -p "$root/logs"'),
    )

  def test_renderer_pins_image_and_hashes_complete_runtime_tree(self):
    renderer = _load_script("render_p1_onehost_stock_admission.py")
    manifest = renderer.render(REPO, "p45", renderer.PINNED_IMAGE_DIGEST)
    self.assertEqual(manifest["execution"]["optimizer_commits"], 0)
    self.assertEqual(manifest["execution"]["mesh"], {"dp": 1, "tp": 4})
    self.assertEqual(
        (
            manifest["execution"]["mini_batch_size"],
            manifest["execution"]["train_micro_batch_size"],
            manifest["execution"]["compute_logps_micro_batch_size"],
        ),
        (1, 1, 1),
    )
    self.assertGreater(manifest["source_tree_files"], 100)
    self.assertEqual(len(manifest["source_tree_sha256"]), 64)
    reconstruction = manifest["source_reconstruction"]
    self.assertEqual(reconstruction["base_commit"], manifest["source_commit"])
    diff = base64.b64decode(
        reconstruction["tracked_diff_base64"], validate=True
    )
    self.assertEqual(hashlib.sha256(diff).hexdigest(), manifest["source_diff_sha256"])
    self.assertEqual(
        reconstruction["untracked_count"], len(reconstruction["untracked"])
    )
    for record in reconstruction["untracked"]:
      payload = base64.b64decode(record["base64"], validate=True)
      self.assertEqual(len(payload), record["bytes"])
      self.assertEqual(hashlib.sha256(payload).hexdigest(), record["sha256"])
    with tempfile.TemporaryDirectory() as tmp:
      rogue = Path(tmp) / "unexpected.txt"
      rogue.write_text("must not enter target evidence", encoding="utf-8")
      with (
          mock.patch.object(
              renderer.subprocess,
              "check_output",
              return_value=b"unexpected.txt\0",
          ),
          self.assertRaisesRegex(ValueError, "unexpected untracked"),
      ):
        renderer._source_reconstruction(Path(tmp), "0" * 40, b"")
    with self.assertRaisesRegex(ValueError, "P0-pinned"):
      renderer.render(REPO, "m15", "sha256:" + "0" * 64)

  def test_launch_preflight_trio_is_closed_and_detects_drift(self):
    preflight = _load_script("preflight_p1_launch.py")
    renderer = _load_script("render_p1_onehost_stock_admission.py")
    real_git = preflight._git

    def admitted_branch(repo, *args):
      if args == ("branch", "--show-current"):
        return preflight.EXPECTED_BRANCH
      return real_git(repo, *args)

    # The runtime branch guard remains unchanged. A unit test of its admitted
    # identity must not depend on the branch hosting the test checkout.
    self.enterContext(mock.patch.object(
        preflight, "_git", side_effect=admitted_branch
    ))
    manifest = renderer.render(REPO, "p45", renderer.PINNED_IMAGE_DIGEST)
    env = {
        **preflight.EXPECTED_ENV,
        admission.SELECTOR_ENV: "p45",
    }
    receipt = preflight.evaluate(REPO, "p45", manifest, env)
    self.assertEqual(receipt["verdict"], "PASS")
    self.assertEqual(receipt["contract_sweep"]["passed"], 9)
    self.assertTrue(receipt["intent_diff"]["manifest_exact"])
    wrong_env = {**env, "CANON_TP_SIZE": "8"}
    failed = preflight.evaluate(REPO, "p45", manifest, wrong_env)
    self.assertEqual(failed["verdict"], "FAIL")
    self.assertIn("resolved_env", failed["failures"])
    drifted = json.loads(json.dumps(manifest))
    drifted["execution"]["optimizer_commits"] = 1
    failed = preflight.evaluate(REPO, "p45", drifted, env)
    self.assertIn("intent_diff", failed["failures"])
    self.assertEqual(failed["intent_diff"]["status"], "FAIL")
    with mock.patch.object(preflight, "_git", return_value="local/wrong-branch"):
      wrong_branch = preflight.evaluate(REPO, "p45", manifest, env)
    self.assertEqual(wrong_branch["verdict"], "FAIL")
    self.assertIn("branch", wrong_branch["intent_diff"]["failures"])

  def test_classifier_requires_every_receipt_and_hbm_headroom(self):
    classifier = _load_script("classify_p1_admission.py")
    renderer = _load_script("render_p1_onehost_stock_admission.py")
    manifest = renderer.render(REPO, "m15", renderer.PINNED_IMAGE_DIGEST)
    snapshot = {
        "checkpoint_revision": admission.CHECKPOINT_REVISION,
        "files": {
            name: {"sha256": expected["sha256"]}
            for name, expected in admission.SNAPSHOT_FILES.items()
        },
    }
    runtime = {
        "workload": "m15", "device_count": 4, "local_device_count": 4,
        "process_count": 1, "platforms": ["tpu"],
        "device_kinds": ["TPU v5"], "device_ids": [0, 1, 2, 3],
        "device_process_indices": [0, 0, 0, 0],
        "jax_version": admission.EXPECTED_JAX_VERSION,
        "rollout_mesh_shape": [1, 4], "trainer_mesh_shape": [1, 4],
    }
    dataset = {
        "workload": "m15",
        "train_sha256": admission.WORKLOADS["m15"]["train_sha256"],
        "eval_sha256": admission.WORKLOADS["m15"]["eval_sha256"],
    }
    tokenizer = {
        "class": "GemmaTokenizer", "vocab_size": 262144,
        "bos_id": 2, "eos_id": 1, "pad_id": 0, "unk_id": 3,
        "chat_template_sha256": admission.CHAT_TEMPLATE_SHA256,
        "token_probe_sha256": admission.TOKEN_PROBE_SHA256,
    }
    hbm = [
        {
            "stage": stage,
            "devices": [{"device_id": index} for index in range(4)],
            "min_bytes_free": 3 * 1024**3,
        }
        for stage in classifier.HBM_STAGES
    ]
    weights = {
        "equal": True, "mismatch_indices": [], "mapped_leaves": 593,
        "live_leaves": 593, "total_elements": 4,
        "mesh_shape": [["data", 1], ["model", 4]],
        "architecture": {
            "layers": 42, "heads": 8, "kv_heads": 2, "head_dim": 256,
            "hidden_size": 2560, "intermediate_size": 10240,
            "vocab_size": 262144, "sliding_window": 512,
            "final_logit_softcapping": 30.0,
            "vocab_size_per_layer_input": 262144,
            "hidden_size_per_layer_input": 256,
            "num_kv_shared_layers": 18, "tie_word_embeddings": True,
            "attention_k_eq_v": False, "enable_moe_block": False,
            "layer_types": list(admission.EXPECTED_LAYER_TYPES),
        },
    }
    optimizer = {"leaves": 4, "elements": 16, "allocated_bytes": 64}
    shape = {
        "workload": "m15", "caller_global_rows": 1, "dp": 1, "tp": 4,
        "mini_batch_size": 1, "train_micro_batch_size": 1,
        "compute_logps_micro_batch_size": 1,
        "context_hard_cap": admission.WORKLOADS["m15"]["context_hard_cap"],
    }
    rollout = {
        "workload": "m15", "trajectories": 1, "prompts": 1,
        "generations": 1, "train_steps_before": 0, "train_steps_after": 0,
        "global_steps_before": 0, "global_steps_after": 0,
        "backward": 0, "optimizer_commits": 0, "max_prompt_tokens": 10,
        "max_assistant_tokens": 2, "max_interactions": 1,
        "max_active_rows": 1, "max_kv_tokens_observed": 12,
    }
    values = {
        "host": [{"hostname": "t1v-n-4a77ebd0-w-0"}],
        "launch_preflight": [{
            "workload": "m15",
            "verdict": "PASS",
            "resolved_env": {"status": "PASS"},
            "contract_sweep": {"status": "PASS"},
            "intent_diff": {"status": "PASS"},
        }],
        "runtime": [runtime], "snapshot": [snapshot],
        "tokenizer": [tokenizer], "dataset": [dataset],
        "hbm": hbm, "optimizer": [optimizer], "weights": [weights],
        "shape": [shape],
        "rollout": [rollout],
    }
    raw = "\n".join(
        prefix + json.dumps(value)
        for name, prefix in classifier.MARKERS.items()
        for value in values[name]
    )
    self.assertEqual(
        classifier.classify(manifest, raw, 0, manifest)["verdict"], "PASS"
    )
    missing = "\n".join(
        line for line in raw.splitlines()
        if not line.startswith(classifier.MARKERS["rollout"])
    )
    failed = classifier.classify(manifest, missing, 0, manifest)
    self.assertEqual(failed["verdict"], "FAIL")
    self.assertIn("rollout_receipt_count=0", failed["failures"])
    changed = {**manifest, "source_tree_sha256": "0" * 64}
    drift = classifier.classify(manifest, raw, 0, changed)
    self.assertIn("runtime_tree_changed_during_run", drift["failures"])
    wrong_weights = {**weights, "mapped_leaves": 592, "total_elements": 0}
    wrong_values = {**values, "weights": [wrong_weights]}
    wrong_raw = "\n".join(
        prefix + json.dumps(value)
        for name, prefix in classifier.MARKERS.items()
        for value in wrong_values[name]
    )
    wrong = classifier.classify(manifest, wrong_raw, 0, manifest)
    self.assertIn("live_weight_leaf_count", wrong["failures"])
    self.assertIn("live_weight_elements", wrong["failures"])
    wrong_shape_values = {
        **values,
        "shape": [{
            **shape,
            "train_micro_batch_size": 2,
            "compute_logps_micro_batch_size": 2,
        }],
    }
    wrong_shape_raw = "\n".join(
        prefix + json.dumps(value)
        for name, prefix in classifier.MARKERS.items()
        for value in wrong_shape_values[name]
    )
    self.assertIn(
        "shape_trainer_batches",
        classifier.classify(
            manifest, wrong_shape_raw, 0, manifest
        )["failures"],
    )
    wrong_tokenizer = {**tokenizer, "token_probe_sha256": "0" * 64}
    wrong_values = {**values, "tokenizer": [wrong_tokenizer]}
    wrong_raw = "\n".join(
        prefix + json.dumps(value)
        for name, prefix in classifier.MARKERS.items()
        for value in wrong_values[name]
    )
    self.assertIn(
        "tokenizer_identity",
        classifier.classify(manifest, wrong_raw, 0, manifest)["failures"],
    )
    reordered_values = {**values, "hbm": list(reversed(hbm))}
    reordered_raw = "\n".join(
        prefix + json.dumps(value)
        for name, prefix in classifier.MARKERS.items()
        for value in reordered_values[name]
    )
    self.assertTrue(any(
        failure.startswith("hbm_stage_order=")
        for failure in classifier.classify(
            manifest, reordered_raw, 0, manifest
        )["failures"]
    ))
    bad_preflight_values = {
        **values,
        "launch_preflight": [{
            "workload": "m15",
            "verdict": "FAIL",
            "resolved_env": {"status": "PASS"},
            "contract_sweep": {"status": "PASS"},
            "intent_diff": {"status": "FAIL"},
        }],
    }
    bad_preflight_raw = "\n".join(
        prefix + json.dumps(value)
        for name, prefix in classifier.MARKERS.items()
        for value in bad_preflight_values[name]
    )
    bad_preflight = classifier.classify(
        manifest, bad_preflight_raw, 0, manifest
    )
    self.assertIn("launch_preflight_verdict", bad_preflight["failures"])
    self.assertIn("launch_preflight_intent_diff", bad_preflight["failures"])
    tampered = json.loads(json.dumps(manifest))
    tampered["source_reconstruction"]["tracked_diff_base64"] = "AA=="
    self.assertIn(
        "source_reconstruction_tracked_diff_sha",
        classifier.classify(tampered, raw, 0, tampered)["failures"],
    )
    tampered = json.loads(json.dumps(manifest))
    tampered["source_reconstruction"]["tracked_diff_sha256"] = "0" * 64
    self.assertIn(
        "source_reconstruction_declared_diff_sha",
        classifier.classify(tampered, raw, 0, tampered)["failures"],
    )
    malformed = raw.replace(
        classifier.MARKERS["rollout"] + json.dumps(rollout),
        classifier.MARKERS["rollout"] + "{not-json}",
    )
    malformed_result = classifier.classify(manifest, malformed, 0, manifest)
    self.assertEqual(malformed_result["verdict"], "FAIL")
    self.assertTrue(any(
        failure.startswith("malformed_receipt:rollout:")
        for failure in malformed_result["failures"]
    ))
    self.assertIn("rollout_receipt_count=0", malformed_result["failures"])


if __name__ == "__main__":
  unittest.main()
