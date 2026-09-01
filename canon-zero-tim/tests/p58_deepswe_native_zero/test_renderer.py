#!/usr/bin/env python3
"""Fail-closed contracts for the paired P58 JobSet renderer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import shlex
import sys
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
sys.path.insert(0, str(PKG / "cluster"))
SPEC = importlib.util.spec_from_file_location(
    "p58_renderer", PKG / "cluster/render_p58_deepswe_tim.py"
)
if SPEC is None or SPEC.loader is None:
  raise RuntimeError("cannot import P58 renderer")
renderer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = renderer
SPEC.loader.exec_module(renderer)


class P58RendererTest(unittest.TestCase):

  def _render(self, arm: str, stage: str = "three-update", **overrides):
    base = overrides.pop(
        "base",
        yaml.safe_load((PKG / "cluster/jobset-64chip.yaml").read_text()),
    )
    kwargs = dict(
        source_commit="1" * 40,
        source_branch="yuxzhang/canon-zero-tim",
        client_image="registry.example/tunix@sha256:" + "2" * 64,
        run_id="pair-test",
        stage=stage,
        arm=arm,
        cpu_nodepool="cpu-np",
        worker_nodepool="tpu-pool",
        model_pvc="model-pvc",
    )
    kwargs.update(overrides)
    return renderer.render(base, **kwargs)

  def test_both_arms_and_horizons_render_on_128_chips(self):
    for arm in ("native", "zero"):
      for stage, steps in (("three-update", 3), ("full", 1000)):
        with self.subTest(arm=arm, stage=stage):
          document = self._render(arm, stage)
          env = renderer.p34._env(document)
          head = renderer.p34._head(document)
          worker = renderer.p34._worker(document)
          network = document["spec"]["network"]
          self.assertIs(head["hostNetwork"], True)
          self.assertEqual(head["dnsPolicy"], "ClusterFirstWithHostNet")
          required_anti_affinity = head["affinity"]["podAntiAffinity"][
              "requiredDuringSchedulingIgnoredDuringExecution"
          ]
          self.assertIn(
              renderer._pathways_head_anti_affinity_term(),
              required_anti_affinity,
          )
          self.assertIs(network["enableDNSHostnames"], True)
          self.assertIs(network["publishNotReadyAddresses"], True)
          self.assertIs(worker["template"]["spec"]["hostNetwork"], True)
          self.assertEqual(
              worker["template"]["spec"]["dnsPolicy"],
              "ClusterFirstWithHostNet",
          )
          self.assertEqual(worker["completions"], 32)
          self.assertEqual(worker["parallelism"], 32)
          self.assertEqual(env["CANON_P58_TIM_ARM"], arm)
          self.assertEqual(env["CANON_P34_DEEPSWE"], "1")
          self.assertEqual(
              document["metadata"]["labels"][
                  renderer._TOKEN_TRANSPORT_LABEL
              ],
              renderer._TOKEN_TRANSPORT,
          )
          self.assertEqual(env["CANON_P58_EXPECTED_UPDATES"], str(steps))
          self.assertIn(f"--max_steps={steps}", env["CANON_RUN_CMD"])
          self.assertIn("--num_generations=16", env["CANON_RUN_CMD"])
          self.assertIn("--seed=42", env["CANON_RUN_CMD"])
          self.assertIn("--max_concurrency=128", env["CANON_RUN_CMD"])
          self.assertNotIn("--max_concurrency=64", env["CANON_RUN_CMD"])
          self.assertIn("--loss_scale_factor=16384", env["CANON_RUN_CMD"])
          args = shlex.split(env["CANON_RUN_CMD"])
          self.assertEqual(
              [item for item in args if item.startswith("--ckpt_dir=")],
              ["--ckpt_dir=none"],
          )
          self.assertFalse(any(
              item.startswith((
                  "--save_interval_steps=", "--max_to_keep="
              ))
              for item in args
          ))
          self.assertIn(
              "--loss_denominator_weighted_accumulation",
              env["CANON_RUN_CMD"],
          )
          self.assertEqual(env["CANON_P34_CLEAN_ROWS"], "1012")
          self.assertEqual(
              env["R2E_K8S_QUEUE_NAME"],
              document["metadata"]["labels"][
                  "kueue.x-k8s.io/queue-name"
              ],
          )
          self.assertEqual(env["R2E_K8S_QUEUE_NAME"], "multislice-queue")
          self.assertEqual(env["NODE_SELECTOR_VAL"], "cpu-np")

  def test_production_render_does_not_reenable_retired_device_probe(self):
    for arm in ("native", "zero"):
      for stage in ("three-update", "full"):
        with self.subTest(arm=arm, stage=stage):
          env = renderer.p34._env(self._render(arm, stage))
          self.assertNotIn(renderer._RETIRED_DEVICE_PROBE_TRIGGER, env)

  def test_retired_device_probe_override_is_rejected(self):
    document = self._render("zero", "full")
    renderer.p34._set_env(
        renderer.p34._container(
            renderer.p34._head(document)["containers"], "jax-tpu"
        ),
        {renderer._RETIRED_DEVICE_PROBE_TRIGGER: "128"},
    )
    with self.assertRaisesRegex(ValueError, "retired Step 65 device probe"):
      renderer.validate(
          document,
          source_commit="1" * 40,
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          stage="full",
          arm="zero",
          worker_nodepool="tpu-pool",
      )

  def test_rollout_timeout_geometry_is_exactly_one_concurrency_wave(self):
    document = self._render("native", "full")
    args = shlex.split(renderer.p34._env(document)["CANON_RUN_CMD"])

    def int_arg(name: str) -> int:
      prefix = f"--{name}="
      values = [int(item.removeprefix(prefix)) for item in args
                if item.startswith(prefix)]
      self.assertEqual(len(values), 1, name)
      return values[0]

    raw_trajectories = int_arg("batch_size") * int_arg("num_generations")
    self.assertEqual(raw_trajectories, 128)
    self.assertEqual(int_arg("max_concurrency"), raw_trajectories)
    self.assertEqual(
        int_arg("rollout_mesh_dp") * int_arg("rollout_vllm_max_num_seqs"),
        raw_trajectories,
    )
    self.assertGreater(
        int_arg("rollout_batch_timeout_secs"),
        int_arg("episode_timeout_secs") + int_arg("cleanup_timeout_secs"),
    )

  def test_fixed_seed_is_unique_and_fail_closed(self):
    document = self._render("native", "full")
    env = renderer.p34._env(document)
    args = shlex.split(env["CANON_RUN_CMD"])
    self.assertEqual(
        [item for item in args if item.startswith("--seed=")],
        ["--seed=42"],
    )
    renderer.p34._set_env(
        renderer.p34._container(
            renderer.p34._head(document)["containers"], "jax-tpu"
        ),
        {
            "CANON_RUN_CMD": env["CANON_RUN_CMD"].replace(
                "--seed=42", "--seed=7"
            )
        },
    )
    with self.assertRaisesRegex(ValueError, "exactly one fixed seed"):
      renderer.validate(
          document,
          source_commit="1" * 40,
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          stage="full",
          arm="native",
          worker_nodepool="tpu-pool",
      )

  def test_checkpoint_disabled_is_unique_and_fail_closed(self):
    document = self._render("zero", "full")
    env = renderer.p34._env(document)
    args = shlex.split(env["CANON_RUN_CMD"])
    self.assertEqual(
        [item for item in args if item.startswith("--ckpt_dir=")],
        ["--ckpt_dir=none"],
    )
    renderer.p34._set_env(
        renderer.p34._container(
            renderer.p34._head(document)["containers"], "jax-tpu"
        ),
        {
            "CANON_RUN_CMD": env["CANON_RUN_CMD"].replace(
                "--ckpt_dir=none", "--ckpt_dir=/tmp/p58-checkpoints"
            )
        },
    )
    with self.assertRaisesRegex(
        ValueError, "requires exactly one --ckpt_dir=none"
    ):
      renderer.validate(
          document,
          source_commit="1" * 40,
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          stage="full",
          arm="zero",
          worker_nodepool="tpu-pool",
      )

  def test_pair_diff_is_registered_treatment_only(self):
    native = self._render("native")
    zero = self._render("zero")
    self.assertEqual(
        renderer.recipe_signature(native), renderer.recipe_signature(zero)
    )
    self.assertEqual(renderer.treatment_signature(native), {
        "arm": "native",
        "alignment_warning_only": "1",
        "proxy_xla": [],
        "high_performance": "0",
        "checked_vma_diagnostic": "",
        "seam_localization": "",
        "disable_sampler_is": "1",
        "disable_tis": "1",
        "sampler_is": (),
    })
    self.assertEqual(renderer.treatment_signature(zero), {
        "arm": "zero",
        "alignment_warning_only": "0",
        "proxy_xla": [renderer.p34.PROXY_XLA_FLAG],
        "high_performance": "0",
        "checked_vma_diagnostic": "",
        "seam_localization": "",
        "disable_sampler_is": "1",
        "disable_tis": "1",
        "sampler_is": (),
    })

  def test_zero_hp_full_is_additive_and_target_only(self):
    document = self._render(
        "zero", "full", high_performance=True, run_id="hp-test"
    )
    env = renderer.p34._env(document)
    self.assertEqual(env["CANON_PROFILE_FILE"], renderer.HP_PROFILE)
    self.assertEqual(env["CANON_V1_HP_FULL"], "1")
    self.assertEqual(env["CANON_P38_FIXED_LM_HEAD"], "1")
    self.assertEqual(env["CANON_DEEPSWE_ALIGNMENT_WARN_ONLY"], "1")
    self.assertIn("--ckpt_dir=none", shlex.split(env["CANON_RUN_CMD"]))
    self.assertEqual(
        document["metadata"]["labels"]["canon.zero-tim/fixed-lm-head"],
        "1",
    )
    self.assertIn("zero-hp-full", document["metadata"]["name"])
    self.assertEqual(env["CANON_P59_CHECKED_VMA"], "1")
    self.assertEqual(env["CANON_P67_P66_VMA_P59_ONLY"], "1")
    self.assertEqual(env["CANON_V1_HP_FIRST_UPDATE_GATE"], "1")
    self.assertEqual(env["CANON_DP_COMPARE_MODE"], "fingerprint-hybrid")
    self.assertEqual(
        env["CANON_DP_DISTINCT_SCHEDULE"], "first-group-warmup"
    )
    self.assertEqual(env["CANON_DP_FINITE_FETCH"], "batched-commit")
    self.assertEqual(env["CANON_P71_SCAN"], "fwd")
    self.assertNotIn("CANON_DP_COLLECTIVE_REDUCE", env)
    for arm, stage in (("native", "full"), ("zero", "three-update")):
      with self.subTest(arm=arm, stage=stage):
        with self.assertRaisesRegex(ValueError, "only for Zero full"):
          self._render(arm, stage, high_performance=True)

  def test_full_zero_hp_tito_identity_is_fail_closed(self):
    document = self._render(
        "zero", "full", high_performance=True, run_id="hp-tito"
    )
    env = renderer.p34._env(document)
    self.assertEqual(env["CANON_P34_DEEPSWE"], "1")
    self.assertEqual(
        document["metadata"]["labels"][renderer._TOKEN_TRANSPORT_LABEL],
        renderer._TOKEN_TRANSPORT,
    )

    document["metadata"]["labels"].pop(renderer._TOKEN_TRANSPORT_LABEL)
    with self.assertRaisesRegex(ValueError, "token transport must be TiTO"):
      renderer.validate(
          document,
          source_commit="1" * 40,
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          stage="full",
          arm="zero",
          worker_nodepool="tpu-pool",
          high_performance=True,
      )

    document = self._render(
        "zero", "full", high_performance=True, run_id="hp-not-deepswe"
    )
    renderer.p34._set_env(
        renderer.p34._container(
            renderer.p34._head(document)["containers"], "jax-tpu"
        ),
        {"CANON_P34_DEEPSWE": "0"},
    )
    with self.assertRaisesRegex(ValueError, "rendered environment mismatch"):
      renderer.validate(
          document,
          source_commit="1" * 40,
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          stage="full",
          arm="zero",
          worker_nodepool="tpu-pool",
          high_performance=True,
      )

  def test_system_optimization_is_absent_from_neighboring_and_diagnostic_arms(self):
    documents = (
        self._render("native", "full"),
        self._render("native", "full", sampler_is=True),
        self._render("zero", "three-update"),
        self._render("zero", "full"),
        self._render(
            "zero", "full", checked_vma_off_diagnostic=True
        ),
        self._render(
            "zero", "full", checked_vma_on_diagnostic=True
        ),
        self._render("zero", "full", seam_localization="coarse"),
    )
    for document in documents:
      env = renderer.p34._env(document)
      with self.subTest(jobset=document["metadata"]["name"]):
        for key in renderer.FULL_SYSTEM_OPTIMIZATION_ENV_NAMES:
          self.assertNotIn(key, env)
        self.assertNotIn("CANON_DP_COLLECTIVE_REDUCE", env)

  def test_checked_vma_off_diagnostic_is_exact_zero_hp_step0_selector(self):
    production = self._render(
        "zero", "full", high_performance=True, run_id="hp-prod"
    )
    diagnostic = self._render(
        "zero",
        "full",
        checked_vma_off_diagnostic=True,
        run_id="vmaoff",
    )
    env = renderer.p34._env(diagnostic)
    self.assertEqual(env["CANON_PROFILE_FILE"], renderer.HP_PROFILE)
    self.assertEqual(env["CANON_V1_HP_FULL"], "1")
    self.assertEqual(env["CANON_P38_FIXED_LM_HEAD"], "1")
    self.assertEqual(
        diagnostic["metadata"]["labels"]["canon.zero-tim/fixed-lm-head"],
        "1",
    )
    self.assertEqual(env["CANON_P58_CHECKED_VMA_DIAGNOSTIC"], "off")
    self.assertEqual(env["CANON_P38_PRECHECK_ONLY"], "1")
    self.assertEqual(env["CANON_P38_CONTROLLED_EXIT"], "1")
    self.assertEqual(env["CANON_P38_DIAGNOSTIC_ROUNDS"], "1")
    self.assertEqual(
        diagnostic["metadata"]["labels"]["canon.zero-tim/backward"], "0"
    )
    self.assertEqual(
        diagnostic["metadata"]["labels"]["canon.zero-tim/optimizer-commits"],
        "0",
    )
    self.assertIn("vmaoff", diagnostic["metadata"]["name"])
    self.assertEqual(
        diagnostic["metadata"]["labels"]["canon.zero-tim/diagnostic"],
        "p58-checked-vma-off",
    )
    self.assertEqual(
        renderer.recipe_signature(production),
        renderer.recipe_signature(diagnostic),
    )
    self.assertEqual(
        renderer.treatment_signature(diagnostic)["checked_vma_diagnostic"],
        "off",
    )

  def test_checked_vma_off_diagnostic_rejects_other_recipes(self):
    for arm, stage, high_performance in (
        ("native", "full", False),
        ("zero", "three-update", False),
        ("zero", "full", True),
    ):
      with self.subTest(
          arm=arm, stage=stage, high_performance=high_performance
      ):
        with self.assertRaisesRegex(ValueError, "its own Zero/full HP"):
          self._render(
              arm,
              stage,
              high_performance=high_performance,
              checked_vma_off_diagnostic=True,
          )

  def test_checked_vma_on_diagnostic_is_matched_zero_commit_control(self):
    off = self._render(
        "zero", "full", checked_vma_off_diagnostic=True, run_id="vmaoff"
    )
    on = self._render(
        "zero", "full", checked_vma_on_diagnostic=True, run_id="vmaon"
    )
    env = renderer.p34._env(on)
    self.assertEqual(env["CANON_P58_CHECKED_VMA_DIAGNOSTIC"], "on")
    self.assertEqual(env["CANON_P38_PRECHECK_ONLY"], "1")
    self.assertEqual(env["CANON_P38_CONTROLLED_EXIT"], "1")
    self.assertEqual(
        on["metadata"]["labels"]["canon.zero-tim/diagnostic"],
        "p58-checked-vma-on",
    )
    self.assertEqual(
        on["metadata"]["labels"]["canon.zero-tim/backward"], "0"
    )
    self.assertEqual(renderer.recipe_signature(off), renderer.recipe_signature(on))
    off_treatment = renderer.treatment_signature(off)
    on_treatment = renderer.treatment_signature(on)
    self.assertEqual(off_treatment["checked_vma_diagnostic"], "off")
    self.assertEqual(on_treatment["checked_vma_diagnostic"], "on")
    off_treatment.pop("checked_vma_diagnostic")
    on_treatment.pop("checked_vma_diagnostic")
    self.assertEqual(off_treatment, on_treatment)

  def test_checked_vma_diagnostic_selectors_are_mutually_exclusive(self):
    with self.assertRaisesRegex(ValueError, "mutually exclusive"):
      self._render(
          "zero",
          "full",
          checked_vma_off_diagnostic=True,
          checked_vma_on_diagnostic=True,
      )

  def test_coarse_seam_is_three_round_frozen_zero_hp_selector(self):
    production = self._render(
        "zero", "full", high_performance=True, run_id="hp-prod"
    )
    diagnostic = self._render(
        "zero", "full", seam_localization="coarse", run_id="seam"
    )
    env = renderer.p34._env(diagnostic)
    self.assertEqual(env["CANON_P58_SEAM_LOCALIZATION"], "coarse")
    self.assertEqual(env["CANON_P38_DIAGNOSTIC_ROUNDS"], "3")
    self.assertEqual(env["CANON_P38_DURABILITY_PROFILE"], "p58-seam-v1")
    self.assertEqual(env["CANON_P38_SEAM_OBSERVER"], "layer")
    self.assertEqual(env["CANON_P38_SEAM_MIN_POSITION"], "1686")
    self.assertEqual(env["CANON_P38_SEAM_MAX_POSITION"], "4096")
    self.assertEqual(env["CANON_P38_SEAM_MAX_BYTES"], "4294967296")
    self.assertEqual(env["CANON_P38_TAIL_OBSERVER"], "1")
    self.assertEqual(env["CANON_P38_MIN_ACTION_KV"], "1686")
    self.assertEqual(
        env["CANON_P38_SERVING_CAPTURE_PREFIX_BOUNDS"],
        "1686,2512,3072,3584,4096",
    )
    for onset in (2513, 3438, 3715, 3880, 4032):
      self.assertLessEqual(int(env["CANON_P38_SEAM_MIN_POSITION"]), onset)
      self.assertLess(onset, int(env["CANON_P38_SEAM_MAX_POSITION"]))
    self.assertEqual(
        diagnostic["metadata"]["labels"]["canon.zero-tim/backward"], "0"
    )
    self.assertEqual(
        diagnostic["metadata"]["labels"]["canon.zero-tim/optimizer-commits"],
        "0",
    )
    self.assertEqual(
        renderer.recipe_signature(production),
        renderer.recipe_signature(diagnostic),
    )
    self.assertEqual(
        renderer.treatment_signature(diagnostic)["seam_localization"],
        "coarse",
    )

  def test_coarse_seam_rejects_foreign_and_checked_vma_recipes(self):
    for kwargs in (
        {"arm": "native", "stage": "full"},
        {"arm": "zero", "stage": "three-update"},
        {"arm": "zero", "stage": "full", "high_performance": True},
        {
            "arm": "zero",
            "stage": "full",
            "checked_vma_on_diagnostic": True,
        },
    ):
      with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
          ValueError, "its own Zero/full HP"
      ):
        arm = kwargs.pop("arm")
        stage = kwargs.pop("stage")
        self._render(arm, stage, seam_localization="coarse", **kwargs)

  def test_checked_vma_prepare_wrapper_is_render_only(self):
    path = (
        PKG
        / "tasks/p58-deepswe-native-zero-comparison/scripts"
        / "prepare_p58_checked_vma_off_diagnostic.sh"
    )
    source = path.read_text()
    self.assertIn("--checked-vma-off-diagnostic", source)
    self.assertIn("--stage full", source)
    self.assertIn("--arm zero", source)
    self.assertIn("origin/yuxzhang/canon-zero-tim", source)
    self.assertNotIn("kubectl ", source)

  def test_checked_vma_aba_wrapper_is_render_only(self):
    path = (
        PKG
        / "tasks/p58-deepswe-native-zero-comparison/scripts"
        / "prepare_p58_checked_vma_aba_wave.sh"
    )
    source = path.read_text()
    self.assertIn("render_p58_checked_vma_aba_wave.py", source)
    self.assertIn("verify_p58_checked_vma_aba_wave.py", source)
    self.assertIn("origin/yuxzhang/canon-zero-tim", source)
    self.assertNotIn("kubectl ", source)

  def test_coarse_seam_prepare_wrapper_is_render_only(self):
    path = (
        PKG
        / "tasks/p58-deepswe-native-zero-comparison/scripts"
        / "prepare_p58_coarse_seam_localization.sh"
    )
    source = path.read_text()
    self.assertIn("--seam-localization coarse", source)
    self.assertIn("--stage full", source)
    self.assertIn("--arm zero", source)
    self.assertNotIn("kubectl ", source)

  def test_optional_algorithm_interventions_are_absent(self):
    for arm in ("native", "zero"):
      args = shlex.split(
          renderer.p34._env(self._render(arm))["CANON_RUN_CMD"]
      )
      self.assertIn("--use_rollout_logps", args)
      self.assertFalse(any(item.startswith("--sampler_is=") for item in args))
      self.assertFalse(
          any(item.startswith("--group_clip_filter_threshold") for item in args)
      )
      self.assertNotIn("--optimizer-offload", args)
      index = args.index("--filter_statuses")
      self.assertEqual(
          tuple(args[index + 1:index + 7]), renderer._FILTER_STATUSES
      )

  def test_native_is_changes_only_the_registered_sampler_recipe(self):
    raw = self._render("native", "full")
    corrected = self._render("native", "full", sampler_is=True)
    raw_env = renderer.p34._env(raw)
    corrected_env = renderer.p34._env(corrected)
    raw_args = shlex.split(raw_env["CANON_RUN_CMD"])
    corrected_args = shlex.split(corrected_env["CANON_RUN_CMD"])

    self.assertNotIn("--sampler_is=token", raw_args)
    self.assertNotIn("--sampler_is_threshold=2.0", raw_args)
    self.assertIn("--sampler_is=token", corrected_args)
    self.assertIn("--sampler_is_threshold=2.0", corrected_args)
    self.assertEqual(raw_env["CANON_P34_DISABLE_SAMPLER_IS"], "1")
    self.assertEqual(raw_env["CANON_P34_DISABLE_TIS"], "1")
    self.assertEqual(corrected_env["CANON_P34_DISABLE_SAMPLER_IS"], "0")
    self.assertEqual(corrected_env["CANON_P34_DISABLE_TIS"], "0")
    self.assertNotIn("canon.zero-tim/sampler-recipe", raw["metadata"]["labels"])
    self.assertEqual(
        corrected["metadata"]["labels"]["canon.zero-tim/sampler-recipe"],
        "token-is",
    )
    self.assertIn("ds4b-native-is-full", corrected["metadata"]["name"])
    for forbidden in ("--group_clip_filter_threshold", "--optimizer-offload"):
      self.assertFalse(
          any(
              item == forbidden or item.startswith(forbidden + "=")
              for item in corrected_args
          )
      )

    ignored = {
        "CANON_RUN_CMD",
        "CANON_RUN_LOG",
        "CANON_STATE",
        "CANON_P34_WEIGHT_REPORT",
        "CANON_PRE_ALIGN_REPORT",
        "CANON_ALIGN_REPORT",
        "CANON_UPDATE_REPORT",
        "CANON_P58_DEBUG_DIR",
        "CANON_WANDB_RUN_NAME",
        "CANON_WANDB_GROUP",
        "CANON_P34_DISABLE_SAMPLER_IS",
        "CANON_P34_DISABLE_TIS",
    }
    self.assertEqual(
        {key: value for key, value in raw_env.items() if key not in ignored},
        {
            key: value
            for key, value in corrected_env.items()
            if key not in ignored
        },
    )
    self.assertEqual(
        renderer.recipe_signature(raw), renderer.recipe_signature(corrected)
    )

  def test_sampler_is_is_rejected_outside_native(self):
    with self.assertRaisesRegex(ValueError, "only for the native arm"):
      self._render("zero", sampler_is=True)
    with self.assertRaisesRegex(ValueError, "only for the native arm"):
      self._render("zero", "full", sampler_is=True, high_performance=True)

  def test_attempt_zero_and_transport_environment_are_exact(self):
    forbidden_env = {
        "PATHWAYS_HEARTBEAT_TIMEOUT_SEC",
        "IFRT_PROXY_TIMEOUT_SECONDS",
        "GRPC_KEEPALIVE_TIME_MS",
        "GRPC_KEEPALIVE_TIMEOUT_MS",
        "GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS",
    }
    for arm in ("native", "zero"):
      document = self._render(arm, "full")
      self.assertEqual(
          document["spec"]["failurePolicy"],
          {"maxRestarts": 0, "restartStrategy": "Recreate"},
      )
      self.assertTrue(forbidden_env.isdisjoint(renderer.p34._env(document)))

  def test_restart_policy_drift_is_rejected(self):
    document = self._render("native", "full")
    document["spec"]["failurePolicy"]["maxRestarts"] = 3
    with self.assertRaisesRegex(ValueError, "Attempt-0"):
      renderer.validate(
          document,
          source_commit="1" * 40,
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          stage="full",
          arm="native",
          worker_nodepool="tpu-pool",
      )

  def test_kueue_managed_pool_uses_jobset_level_exclusive_topology(self):
    for worker_nodepool in ("auto", "none", "tpu-v5p-slice", "any"):
      with self.subTest(worker_nodepool=worker_nodepool):
        document = self._render(
            "native", "full", worker_nodepool=worker_nodepool
        )
        self.assertEqual(
            document["metadata"]["annotations"][
                renderer._EXCLUSIVE_TOPOLOGY_ANNOTATION
            ],
            "cloud.google.com/gke-nodepool",
        )
        worker = renderer.p34._worker(document)
        worker_metadata = worker["template"].get("metadata", {})
        self.assertNotIn(
            renderer._EXCLUSIVE_TOPOLOGY_ANNOTATION,
            worker_metadata.get("annotations", {}),
        )
        self.assertNotIn(
            "cloud.google.com/gke-nodepool",
            worker["template"]["spec"]["nodeSelector"],
        )

  def test_explicit_worker_pool_remains_exact(self):
    document = self._render(
        "native", "full", worker_nodepool="mlperf-v5p-128-np-0"
    )
    worker_pod = renderer.p34._worker(document)["template"]["spec"]
    self.assertEqual(
        worker_pod["nodeSelector"]["cloud.google.com/gke-nodepool"],
        "mlperf-v5p-128-np-0",
    )

  def test_exclusive_topology_annotation_scope_is_fail_closed(self):
    document = self._render("native", "full", worker_nodepool="auto")
    document["metadata"]["annotations"].pop(
        renderer._EXCLUSIVE_TOPOLOGY_ANNOTATION
    )
    with self.assertRaisesRegex(ValueError, "lost its exclusive-topology"):
      renderer.validate(
          document,
          source_commit="1" * 40,
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          stage="full",
          arm="native",
          worker_nodepool="auto",
      )

    document = self._render("native", "full", worker_nodepool="auto")
    worker = renderer.p34._worker(document)
    worker["template"].setdefault("metadata", {}).setdefault(
        "annotations", {}
    )[renderer._EXCLUSIVE_TOPOLOGY_ANNOTATION] = (
        "cloud.google.com/gke-nodepool"
    )
    with self.assertRaisesRegex(ValueError, "must not be on the Pod template"):
      renderer.validate(
          document,
          source_commit="1" * 40,
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          stage="full",
          arm="native",
          worker_nodepool="auto",
      )

  def test_unregistered_data_arm_and_stage_are_rejected(self):
    with self.assertRaisesRegex(ValueError, "native or zero"):
      self._render("mystery")
    with self.assertRaisesRegex(ValueError, "three-update or full"):
      self._render("native", "one-update")
    with self.assertRaisesRegex(ValueError, "1012-task"):
      self._render(
          "native", whitelist="unreviewed.jsonl", whitelist_sha256="3" * 64
      )

  def test_missing_or_invalid_parent_queue_is_rejected(self):
    for queue_name in (None, "Bad Queue"):
      with self.subTest(queue_name=queue_name):
        base = yaml.safe_load(
            (PKG / "cluster/jobset-64chip.yaml").read_text()
        )
        labels = base["metadata"]["labels"]
        if queue_name is None:
          labels.pop("kueue.x-k8s.io/queue-name")
        else:
          labels["kueue.x-k8s.io/queue-name"] = queue_name
        with self.assertRaisesRegex(ValueError, "exact Kueue LocalQueue"):
          self._render("native", base=base)

  def test_nonadmitted_cpu_nodepool_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "admitted cpu-np"):
      self._render("native", cpu_nodepool="deepswe-cpu-pool")

  def test_head_host_network_regression_is_rejected(self):
    document = self._render("native", "full")
    head = renderer.p34._head(document)
    head["hostNetwork"] = False
    head["dnsPolicy"] = "ClusterFirst"
    with self.assertRaisesRegex(ValueError, "retain the Pathways host network"):
      renderer.validate(
          document,
          source_commit="1" * 40,
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          stage="full",
          arm="native",
          worker_nodepool="tpu-pool",
      )

  def test_jobset_dns_regression_is_rejected(self):
    document = self._render("native", "full")
    document["spec"]["network"]["enableDNSHostnames"] = False
    with self.assertRaisesRegex(ValueError, "requires JobSet Pod DNS"):
      renderer.validate(
          document,
          source_commit="1" * 40,
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          stage="full",
          arm="native",
          worker_nodepool="tpu-pool",
      )

  def test_pathways_head_anti_affinity_regression_is_rejected(self):
    document = self._render("native", "full")
    head = renderer.p34._head(document)
    head["affinity"]["podAntiAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ] = []
    with self.assertRaisesRegex(ValueError, "required Pathways anti-affinity"):
      renderer.validate(
          document,
          source_commit="1" * 40,
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          stage="full",
          arm="native",
          worker_nodepool="tpu-pool",
      )

  def test_worker_resource_manager_address_regression_is_rejected(self):
    document = self._render("native", "full")
    worker_pod = renderer.p34._worker(document)["template"]["spec"]
    worker = renderer.p34._container(
        worker_pod["containers"], "pathways-worker"
    )
    worker["args"] = [
        "--resource_manager_address=foreign-head:29001"
        if item.startswith("--resource_manager_address=") else item
        for item in worker["args"]
    ]
    with self.assertRaisesRegex(ValueError, "resource-manager address"):
      renderer.validate(
          document,
          source_commit="1" * 40,
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          stage="full",
          arm="native",
          worker_nodepool="tpu-pool",
      )

  def test_worker_host_network_regression_is_rejected(self):
    document = self._render("native", "full")
    worker_pod = renderer.p34._worker(document)["template"]["spec"]
    worker_pod["hostNetwork"] = False
    worker_pod["dnsPolicy"] = "ClusterFirst"
    with self.assertRaisesRegex(ValueError, "retain the host network"):
      renderer.validate(
          document,
          source_commit="1" * 40,
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          stage="full",
          arm="native",
          worker_nodepool="tpu-pool",
      )

  def test_worker_pathways_head_regression_is_rejected(self):
    document = self._render("native", "full")
    worker_pod = renderer.p34._worker(document)["template"]["spec"]
    worker = renderer.p34._container(
        worker_pod["containers"], "pathways-worker"
    )
    for item in worker["env"]:
      if item["name"] == "PATHWAYS_HEAD":
        item["value"] = "foreign-head"
    with self.assertRaisesRegex(ValueError, "JobSet Pod DNS name"):
      renderer.validate(
          document,
          source_commit="1" * 40,
          client_image="registry.example/tunix@sha256:" + "2" * 64,
          stage="full",
          arm="native",
          worker_nodepool="tpu-pool",
      )

  def test_custom_instance_type_is_accepted_and_validated(self):
    document = self._render("zero", "full", instance_type="4x4x8_nowrap")
    head = renderer.p34._head(document)
    services = renderer._service_containers(head)
    manager = renderer.p34._container(services, "pathways-rm")
    self.assertIn("--instance_type=tpuv5:4x4x8_nowrap", manager["args"])
    worker_pod = renderer.p34._worker(document)["template"]["spec"]
    worker = renderer.p34._container(
        worker_pod["containers"], "pathways-worker"
    )
    self.assertIn("--instance_type=tpuv5:4x4x8_nowrap", worker["args"])
    renderer.validate(
        document,
        source_commit="1" * 40,
        client_image="registry.example/tunix@sha256:" + "2" * 64,
        stage="full",
        arm="zero",
        worker_nodepool="tpu-pool",
        instance_type="4x4x8_nowrap",
    )


if __name__ == "__main__":
  unittest.main()
