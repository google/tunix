"""Fail-closed contracts for the FrozenLake four-chip one-host carriers."""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import importlib.util
import json
import os
import pathlib
from pathlib import Path
import shlex
import subprocess
import sys
import unittest
from unittest import mock

from tunix.rl import dp_workloads


ROOT = Path(__file__).resolve().parents[3]
TRAIN_ENTRYPOINT = ROOT / "examples/frozenlake/train_frozenlake_qwen3.py"
MODEL_DIR = ROOT / "canon-zero-tim/src/engine_shims/models/qwen8b_tp2"
PROFILE = (
    ROOT
    / "canon-zero-tim/cluster/profiles/"
    "qwen3-8b-dp2-tp2-frozenlake-onehost.env"
)
PROFILE_REL = "cluster/profiles/qwen3-8b-dp2-tp2-frozenlake-onehost.env"


def _source_matrix_profile(
    *,
    dp_size: int,
    tp_size: int,
    recipe: str,
    keep_tape: str,
    reduce_once: str,
    length_sort: str,
    overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
  profile = (
      "cluster/profiles/qwen3-8b-"
      f"dp{dp_size}-tp{tp_size}-frozenlake-onehost.env"
  )
  values = {
      "CANON_PROFILE_FILE": profile,
      "CANON_P32_KEEP_TAPE": keep_tape,
      "CANON_DP_REDUCE_ONCE": reduce_once,
      "CANON_P32_LENGTH_SORT": length_sort,
      "CANON_P75_REPORT_ADJOINT_BUCKETS": "0",
      "CANON_P76_CHUNK_DEPENDENCY_TICKET": "0",
      "CANON_P77_CHUNK_BACKPRESSURE": "0",
      "CANON_P78_SEGMENTED_ACTOR_LOGPS": (
          "1" if recipe == "p45" and dp_size == 4 else "0"
      ),
      "CANON_P57_WORKLOAD_CANDIDATE": "m15" if recipe == "m15" else "",
      "CANON_P57_DATA_SPLIT": "main" if recipe == "m15" else "",
      "CANON_P57_RUN_KIND": "",
      "CANON_P57_TIM_ARM": "",
      "XLA_FLAGS": "--xla_allow_excess_precision=false",
  }
  values.update(overrides or {})
  exports = "\n".join(
      f"export {key}={shlex.quote(value)}" for key, value in values.items()
  )
  script = f"""
set -euo pipefail
{exports}
export JAX_PLATFORMS=cpu
export PYTHONPYCACHEPREFIX=/tmp/v2fl_profile_pycache
source {shlex.quote('canon-zero-tim/' + profile)}
{shlex.quote(sys.executable)} -c {shlex.quote(
    "import os; from tunix.rl import dp_workloads; "
    "workload = dp_workloads.get_workload(os.environ['CANON_P32_WORKLOAD']); "
    "dp_workloads.validate_environment(workload, os.environ, "
    "require_reduction_admission=True); print('PROFILE_CONTRACT_PASS')"
)}
printf '%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
  "$CANON_P32_WORKLOAD" "$CANON_MODEL_DIR_NAME" \
  "$CANON_DP_SIZE" "$CANON_TP_SIZE" \
  "$CANON_P32_KEEP_TAPE" "$CANON_DP_REDUCE_ONCE" \
  "$CANON_P32_LENGTH_SORT" "$CANON_P66_P59_CHECK_VMA" \
  "$CANON_P67_P66_VMA_P59_ONLY" \
  "$CANON_EXPECT_MODEL_MESH_IDS" "$CANON_EXPECT_TRAIN_MESH_IDS" \
  "$FL_VLLM_HBM_UTIL"
"""
  return subprocess.run(
      ["bash", "-c", script],
      cwd=ROOT,
      text=True,
      capture_output=True,
      env={},
      check=False,
  )


def _load_contract():
  name = "v2_qwen8b_tp2_contract"
  spec = importlib.util.spec_from_file_location(
      name, MODEL_DIR / "p22xf_contract.py"
  )
  if spec is None or spec.loader is None:
    raise RuntimeError("cannot import Qwen3-8B TP2 projection contract")
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  spec.loader.exec_module(module)
  return module


CONTRACT = _load_contract()


def _environment(workload_name: str) -> dict[str, str]:
  workload = dp_workloads.get_workload(workload_name)
  candidate = "m15" if workload_name.startswith("frozenlake-m15-") else ""
  profile = (
      "cluster/profiles/qwen3-8b-"
      f"dp{workload.dp_size}-tp{workload.tp_size}-frozenlake-onehost.env"
  )
  split = "main" if candidate else ""
  return {
      "CANON_PROFILE_FILE": profile,
      "CANON_MODEL_DIR_NAME": workload.model_dir_name,
      "CANON_P32_WORKLOAD": workload.name,
      "CANON_P32_TRAIN_ADMITTED": "1",
      "CANON_P32_DP_REDUCTION_ADMITTED": "1",
      "CANON_P33_WORKLOAD_LAUNCH_ADMITTED": "1",
      "CANON_P33_NO_COMMIT": "1",
      "CANON_P33_RUN_STAGE": "backward-no-commit",
      "CANON_P57_WORKLOAD_CANDIDATE": candidate,
      "CANON_P57_DATA_SPLIT": split,
      "CANON_P59_RANK_PARALLEL_BACKWARD": "1",
      "CANON_P66_P59_CHECK_VMA": "1" if workload.tp_size > 1 else "0",
      "CANON_P59_CHECKED_VMA": "0",
      "CANON_V1_HP_FULL": "0",
      "CANON_FROZENLAKE_ALIGNMENT_WARN_ONLY": "0",
      "CANON_DP_SIZE": str(workload.dp_size),
      "CANON_TP_SIZE": str(workload.tp_size),
      "CANON_TOTAL_DEVICES": "4",
      "CANON_ENGINE_DP_SIZE": str(workload.dp_size),
      "CANON_QWEN3_TP_SIZE": str(workload.tp_size),
      "CANON_GLOBAL_PROMPTS": "4",
      "CANON_LOCAL_PROMPTS": str(workload.local_prompts),
      "CANON_NUM_GENERATIONS": "4",
      "CANON_LOCAL_TRAJECTORIES": str(workload.local_trajectories),
      "CANON_GLOBAL_TRAJECTORIES": "16",
      "CANON_LOGPROB_M": "256",
      "MIN_TOKEN_BUCKET": str(workload.global_m),
      "CANON_FIXED_AR": "1",
      "CANON_FIXED_AR_EMBED": "1",
      "CANON_RPA_VJP2": "1",
      "CANON_VJP2_MAX_SEQS": "1",
      "CANON_PROMPT_PROCESSED_LOGPROBS": "1",
      "CANON_PALLAS_LOGSOFTMAX": "1",
      "CANON_P32_DP16_SEGMENTED": "1",
      "CANON_P28_SEGMENTED_FORWARD": "1",
      "CANON_P28_SEGMENTED_TRAIN": "1",
      "CANON_P28_G6_UPDATE": "1",
      "CANON_P29_FULL_TRAIN": "1",
      "CANON_ALIGNMENT_GATE": "1",
      "CANON_ALIGNMENT_GATE_ONLY": "0",
      "CANON_ALIGNMENT_UPDATE_CANARY": "0",
      "CANON_ALIGNMENT_TRAIN": "1",
      "CANON_PRE_ALIGN_GATE": "1",
      "CANON_P33_SHORT_ALIGNMENT": "0",
      "CANON_OPT_STATE_RESIDENT": "0",
      "CANON_P30_OPT_STATE_OFFLOAD": "1",
      "CANON_P30_SPARSE_GRAD_ASSEMBLY": "1",
      "CANON_P30_FUSED_PAIR_ACCUMULATION": "0",
      "CANON_P30_REUSE_SEGMENTED_ENGINE": "1",
      "CANON_P30_RELEASE_CAPTURED_STATE": "1",
      "CANON_P30_RESHARD_ACCUMULATOR": "1",
      "FL_SHARED_MESH": f"{workload.dp_size},{workload.tp_size}",
      "CANON_P33_ENABLE_EVAL": "0",
      "CANON_P33_DISABLE_EVAL": "1",
      "CANON_P31_ENABLE_EVAL": "0",
      "CANON_DP_COMPARE_MODE": "fingerprint-hybrid",
      "CANON_DP_DISTINCT_SCHEDULE": "first-group-warmup",
      "CANON_DP_FINITE_FETCH": "batched-commit",
      "CANON_P71_SCAN": "fwd",
      "CANON_P75_REPORT_ADJOINT_BUCKETS": "0",
      "CANON_P76_CHUNK_DEPENDENCY_TICKET": "0",
      "CANON_P77_CHUNK_BACKPRESSURE": "0",
      "CANON_P78_SEGMENTED_ACTOR_LOGPS": (
          "1"
          if workload_name == "frozenlake-p45-onehost-dp4-tp1"
          else "0"
      ),
      "CANON_WANDB_ONLINE_REQUIRED": "0",
      "CANON_P31_MONOTONIC_METRICS": "1",
      "CANON_WANDB_PROJECT": workload.wandb_project,
      "CANON_WANDB_GROUP": "test-frozenlake-onehost",
      "CANON_WANDB_RUN_NAME": "test-frozenlake-onehost",
      "WANDB_MODE": "disabled",
      "XLA_FLAGS": (
          "--xla_cpu_max_isa=AVX2 --xla_allow_excess_precision=false"
      ),
  }


class FrozenLakeOneHostContractTest(unittest.TestCase):

  def test_qwen8b_tp2_projection_shapes_are_exact(self):
    CONTRACT.validate_manifest(CONTRACT.SITES)
    self.assertEqual(CONTRACT.TP_SIZE, 2)
    self.assertEqual((CONTRACT.BM, CONTRACT.BN, CONTRACT.BK), (128, 256, 256))
    self.assertEqual(CONTRACT.MATMUL_N_PADDING, {75968: 76032})
    self.assertEqual(
        {(site.family, site.k_local, site.n_local) for site in CONTRACT.SITES},
        {
            ("q_proj", 4096, 2048),
            ("k_proj", 4096, 512),
            ("v_proj", 4096, 512),
            ("o_proj", 2048, 4096),
            ("gate_proj", 4096, 6144),
            ("up_proj", 4096, 6144),
            ("down_proj", 6144, 4096),
        },
    )

  def test_qwen8b_tp2_rejects_other_tp_environments(self):
    values = {
        "CANON_QWEN3_HIDDEN_SIZE": "4096",
        "CANON_QWEN3_INTERMEDIATE_SIZE": "12288",
        "CANON_QWEN3_NUM_ATTENTION_HEADS": "32",
        "CANON_QWEN3_NUM_KV_HEADS": "8",
        "CANON_QWEN3_HEAD_DIM": "128",
        "CANON_QWEN3_TP_SIZE": "4",
        "CANON_PALLAS_ALL_PROJ": "1",
        "CANON_FIXED_AR": "1",
    }
    with mock.patch.dict(os.environ, values, clear=True):
      with self.assertRaisesRegex(RuntimeError, "CANON_QWEN3_TP_SIZE='4'"):
        CONTRACT.preflight(require_enabled=True)

  def test_qwen8b_tp2_manifest_reuses_only_the_reviewed_8b_wrapper(self):
    for line in (MODEL_DIR / "MANIFEST.sha256").read_text().splitlines():
      digest, name = line.split()
      source = (
          ROOT / "canon-zero-tim/src/engine_shims/models/qwen8b/qwen3_p22xh.py"
          if name == "qwen3_p22xh.py"
          else MODEL_DIR / name
      )
      self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)
    installer = (ROOT / "canon-zero-tim/install.sh").read_text()
    self.assertIn(
        'if [ "$MODEL" = qwen8b_tp1 ] || [ "$MODEL" = qwen8b_tp2 ]; then',
        installer,
    )
    probe = (
        ROOT
        / "canon-zero-tim/tests/v2_frozenlake_onehost/"
        "probe_qwen8b_tp2_overlay.py"
    ).read_text()
    self.assertIn('"CANON_QWEN3_TP_SIZE": "2"', probe)
    self.assertIn("V2_QWEN8B_TP2_IMPORT_PASS", probe)

  def test_p22xh_layer_prefix_affects_only_site_suffix_and_diagnostics(self):
    wrapper = (
        ROOT / "canon-zero-tim/src/engine_shims/models/qwen8b/qwen3_p22xh.py"
    )
    tree = ast.parse(wrapper.read_text(encoding="utf-8"))
    site_definition = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_site"
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[site_definition], type_ignores=[]),
            str(wrapper),
            "exec",
        ),
        namespace,
    )
    expected_sites = {
        "input_layernorm": "input",
        "post_attention_layernorm": "post",
        "q_norm": "q",
        "k_norm": "k",
    }
    for suffix, expected in expected_sites.items():
      self.assertEqual(
          namespace["_site"](f"model.layers.0.{suffix}"), expected
      )
      self.assertEqual(
          namespace["_site"](f"model.layers.35.{suffix}"), expected
      )
    with self.assertRaisesRegex(RuntimeError, "unregistered RMSNorm prefix"):
      namespace["_site"]("model.layers.0.not_a_norm")

    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    site_control_reads = 0
    prefix_attributes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "_p22xh_prefix"
    ]
    self.assertGreater(len(prefix_attributes), 1)
    for node in prefix_attributes:
      if isinstance(node.ctx, ast.Store):
        continue
      parent = parents[node]
      if isinstance(parent, ast.FormattedValue):
        continue
      if (
          isinstance(parent, ast.Call)
          and isinstance(parent.func, ast.Name)
          and parent.func.id == "_site"
      ):
        site_control_reads += 1
        continue
      self.fail(
          "_p22xh_prefix gained an execution use outside _site: "
          f"{ast.dump(parent)}"
      )
    self.assertEqual(site_control_reads, 1)

  def test_onehost_proxy_pins_vllm_seed_without_impersonating_p57(self):
    source = TRAIN_ENTRYPOINT.read_text(encoding="utf-8")
    initialization = source.index("frozenlake_onehost_proxy = False")
    canonical_branch = source.index("if CANON_P32_WORKLOAD:", initialization)
    self.assertLess(initialization, canonical_branch)
    self.assertIn(
        "if CANON_P57_RUN_KIND or frozenlake_onehost_proxy", source
    )
    self.assertIn("[V2.FL.SEED] CONTRACT_PASS", source)
    inner = (
        ROOT
        / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts/"
        "run_frozenlake_dp2tp2_inner.sh"
    ).read_text(encoding="utf-8")
    self.assertIn('"data_shuffle_seed": 42', inner)
    self.assertIn('"vllm_global_seed": 0', inner)
    self.assertIn('"vllm_hbm_utilization"', inner)

  def test_onehost_proxy_uses_only_the_disabled_local_wandb_attestor(self):
    source = TRAIN_ENTRYPOINT.read_text(encoding="utf-8")
    self.assertIn(
        "dp_workloads.require_workload_wandb_run(P32_WORKLOAD)", source
    )
    self.assertIn("[V2.FL.WANDB] DISABLED_LOCAL_PASS", source)
    self.assertIn("[CANON_P33_WANDB] ONLINE_RUN_PASS", source)
    runner = (
        ROOT
        / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts/"
        "run_frozenlake_dp2tp2_onehost.sh"
    ).read_text(encoding="utf-8")
    self.assertNotIn("WANDB_API_KEY", runner)

  def test_onehost_proxy_has_an_exact_no_is_alignment_receipt(self):
    learner = (
        ROOT / "tunix/rl/agentic/agentic_grpo_learner.py"
    ).read_text(encoding="utf-8")
    self.assertIn("_v2_frozenlake_onehost_alignment_enabled", learner)
    self.assertIn("[V2.FL.SAMPLER] CONTRACT_PASS", learner)
    self.assertIn("cell in admitted_cells", learner)
    self.assertIn('"0" if tp == 1 else "1"', learner)

  def test_beta_zero_elides_only_an_unconsumed_reference_model(self):
    source = TRAIN_ENTRYPOINT.read_text(encoding="utf-8")
    learner = (
        ROOT / "tunix/rl/agentic/agentic_grpo_learner.py"
    ).read_text(encoding="utf-8")
    self.assertIn("FORCE_COMPUTE_KL = False", source)
    self.assertIn("REFERENCE_MODEL_REQUIRED = requires_reference_model(", source)
    self.assertIn("if REFERENCE_MODEL_REQUIRED", source)
    self.assertIn("else None", source)
    self.assertIn("force_compute_kl=FORCE_COMPUTE_KL", source)
    self.assertIn("[V2.FL.REFERENCE] ADMISSION_PASS", source)
    self.assertEqual(learner.count("if requires_reference_model("), 2)
    self.assertIn("def requires_reference_model(", learner)
    self.assertIn("return force_compute_kl or beta != 0.0", learner)

  def test_p61_full_tree_capture_is_a_measure_only_selector(self):
    """V2_FL_CAPTURE_FULL_TREE=1 hands the learner the P61 capture root.

    The Qwen3-8B fp64 re-pin (tasks/v2_integrate Phase B) captures the
    no-commit proxy's model_before/example/gradient/logps trees; the learner
    admits the FrozenLake proxy and captures before the no-commit discard.
    """
    root = pathlib.Path(__file__).resolve().parents[3]
    onehost = (
        root / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts"
        / "run_frozenlake_dp2tp2_onehost.sh"
    ).read_text(encoding="utf-8")
    self.assertIn('capture_full_tree="${V2_FL_CAPTURE_FULL_TREE:-0}"', onehost)
    self.assertIn(
        'if [ "$capture_full_tree" = 1 ] && [ "$mode" != measure ]; then',
        onehost,
    )
    self.assertIn(
        '-e CANON_P61_BACKWARD_NUMERICAL_DIR="$([ "$capture_full_tree" = 1 ]'
        ' && echo "$root/p61_numerical")"',
        onehost,
    )
    learner = (
        root / "tunix/rl/agentic/agentic_rl_learner.py"
    ).read_text(encoding="utf-8")
    self.assertIn("p61_frozenlake_no_commit_admission = (", learner)
    self.assertIn(
        'and run_stage == "backward-no-commit"\n          and p33_no_commit',
        learner,
    )
    capture = learner.index("if p61_capture_dir and p33_no_commit:")
    discard = learner.index("if p33_no_commit and reduce_once_accumulator_adopted:")
    self.assertLess(capture, discard)

  def test_hbm_stage_diagnostic_is_measure_r0_and_r0b_only(self):
    learner = (
        ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
    ).read_text(encoding="utf-8")
    adapter = (
        ROOT / "tunix/rl/canonical_qwen3_adapter.py"
    ).read_text(encoding="utf-8")
    inner = (
        ROOT
        / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts/"
        "run_frozenlake_dp2tp2_inner.sh"
    ).read_text(encoding="utf-8")
    self.assertIn("workload.frozenlake_four_chip_proxy", learner)
    self.assertIn('os.environ.get("V2_FL_MODE", "") == "measure"', learner)
    self.assertIn(
        'os.environ.get("V2_FL_ARM", "") in ("r0", "r0b")', learner
    )
    self.assertIn('segmented_kwargs["hbm_stage_sink"]', learner)
    self.assertIn('hbm_stage_boundary("after_model_backward"', adapter)
    self.assertIn('hbm_stage_boundary("after_reduce_compare"', adapter)
    self.assertIn('hbm_stage_boundary("after_sink_delete"', adapter)
    self.assertIn("emit_memory_analysis=True", adapter)
    self.assertIn("[V2.FL.REPORT_ADJOINT_MEMORY]", adapter)
    self.assertIn('hbm_chunk_stage_sink("before"', adapter)
    self.assertIn('"after_pullbacks", chunk_index', adapter)
    self.assertIn('hbm_chunk_stage_sink("after_accumulate"', adapter)
    self.assertIn('hbm_stage_sink("report_group_after_bucket_0"', adapter)
    self.assertIn('bucket == 0 and stage == "after_execute"', adapter)
    self.assertIn("hbm_stage_group = 1", adapter)
    self.assertIn('key=lambda index: int(specs[index]["num_chunks"])', adapter)
    tree = ast.parse(adapter)
    callback_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_p32_reverse_group"
    ]
    callback_keywords = [
        keyword.value
        for node in callback_calls
        for keyword in node.keywords
        if keyword.arg == "hbm_chunk_stage_sink"
    ]
    self.assertEqual(len(callback_keywords), 1)
    self.assertIsInstance(callback_keywords[0], ast.Lambda)
    self.assertTrue(any(
        not any(
            keyword.arg == "hbm_chunk_stage_sink" for keyword in node.keywords
        )
        for node in callback_calls
    ))
    self.assertIn('os.environ["V2_FL_MODE"] == "measure"', inner)
    self.assertIn(
        'os.environ["V2_FL_ARM"] in ("r0", "r0b")', inner
    )

  def test_reduce_once_lazy_accumulator_is_fail_closed_in_the_learner(self):
    learner = (
        ROOT / "tunix/rl/agentic/agentic_rl_learner.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(learner)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
    }
    update_functions = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "_run_p28_g6_update"
    ]
    self.assertEqual(len(update_functions), 1)
    segmented_calls = [
        node
        for node in ast.walk(update_functions[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "segmented_dp_grpo_value_and_grad"
    ]

    self.assertIn("loan_precomputed_gradient_accumulator", called_attributes)
    self.assertIn("lazy_reduce_once_accumulator_enabled", called_attributes)
    self.assertIn("adopt_precomputed_scaled_gradient", called_attributes)
    self.assertIn(
        "discard_adopted_precomputed_gradients", called_attributes
    )
    self.assertIn("dp_reduce_once_mode", called_attributes)
    self.assertEqual(len(segmented_calls), 1)
    self.assertIn(
        'segmented_kwargs["gradient_accumulator_loan"]', learner
    )
    self.assertIn('os.environ.get("V2_P0_NEGATIVE_CONTROL", "")', learner)
    self.assertIn(
        "reduce-once returned without adopting its accumulator storage", learner
    )
    self.assertIn("[V2.REDUCE_ONCE.ACCUMULATOR_RESET]", learner)
    self.assertNotIn("not p33_no_commit or numeric_debug", learner)

  def test_reduce_once_adoption_keeps_the_established_scaled_norm_graph(self):
    source = (
        ROOT / "tunix/sft/peft_trainer.py"
    ).read_text(encoding="utf-8")
    methods = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef)
        and node.name == "_precomputed_gradient_adopt_scaled_step"
    ]
    self.assertEqual(len(methods), 1)
    norm_calls = [
        node
        for node in ast.walk(methods[0])
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_precomputed_gradient_norm"
    ]
    self.assertEqual(len(norm_calls), 1)
    self.assertEqual(len(norm_calls[0].args), 1)
    self.assertIsInstance(norm_calls[0].args[0], ast.Name)
    self.assertEqual(norm_calls[0].args[0].id, "scaled")

  def test_p45_r2_anchor_pins_the_deliberate_repin_receipt(self):
    registry = json.loads((
        ROOT
        / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts/"
        "gradient_anchors.json"
    ).read_text(encoding="utf-8"))
    self.assertEqual(
        registry["anchors"].get("p45:dp2-tp2:r2"),
        {
            "run_id": "v2fl_p45_r0d_capsule_20260904_r32",
            "training_capsule_sha256": (
                "99b6dcaba5b816644a02037ef8f4e8ae"
                "0199eb1106a4142e3d076b3f48d8539c"
            ),
            "micro_gradient_norms": [
                37.869327545166016,
                6.235235691070557,
                6.509113311767578,
                7.796840667724609,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "update_gradient_norm": 23.40300178527832,
        },
    )

  def test_p45_r3_anchor_pins_the_length_sorted_receipt(self):
    registry = json.loads((
        ROOT
        / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts/"
        "gradient_anchors.json"
    ).read_text(encoding="utf-8"))
    self.assertEqual(
        registry["anchors"].get("p45:dp2-tp2:r3"),
        {
            "run_id": "v2fl_p45_r0d_capsule_20260904_r32",
            "training_capsule_sha256": (
                "99b6dcaba5b816644a02037ef8f4e8ae"
                "0199eb1106a4142e3d076b3f48d8539c"
            ),
            "micro_gradient_norms": [
                0.0,
                6.509113311767578,
                9.983430862426758,
                0.0,
                0.0,
                0.0,
                0.0,
                37.86933135986328,
            ],
            "update_gradient_norm": 23.40300178527832,
        },
    )

  def test_onehost_proxy_preserves_the_three_epoch_dataset_capacity(self):
    source = TRAIN_ENTRYPOINT.read_text(encoding="utf-8")
    self.assertIn("NUM_EPOCHS = 3", source)
    self.assertIn(
        "if available_updates != 450 or MAX_STEPS > available_updates:",
        source,
    )
    self.assertNotIn("expected_available_updates", source)

  def test_p45_and_m15_proxy_workloads_have_exact_distinct_envelopes(self):
    for recipe, response, turns, candidate_args in (
        ("p45", 2048, 5, ()),
        (
            "m15",
            8192,
            15,
            ("--p57_workload_candidate=m15", "--p57_data_split=main"),
        ),
    ):
      for dp_size, tp_size, model_dir in (
          (4, 1, "qwen8b_tp1"),
          (2, 2, "qwen8b_tp2"),
          (1, 4, "qwen8b"),
      ):
        name = f"frozenlake-{recipe}-onehost-dp{dp_size}-tp{tp_size}"
        with self.subTest(name=name):
          prompt = 4096
          workload = dp_workloads.get_workload(name)
          self.assertEqual(
              (workload.dp_size, workload.tp_size), (dp_size, tp_size)
          )
          self.assertEqual(workload.model_id, "Qwen/Qwen3-8B")
          self.assertEqual(workload.model_dir_name, model_dir)
          self.assertEqual(workload.global_trajectories, 16)
          self.assertEqual(workload.gradient_groups, 16 // dp_size)
          self.assertEqual(workload.global_m, dp_size * 256)
          self.assertEqual(
              (workload.max_prompt_length, workload.max_response_length),
              (prompt, response),
          )
          self.assertEqual(workload.frozenlake_max_turns, turns)
          command = workload.command(run_stage="backward-no-commit")
          for argument in (
              f"--mesh_dp={dp_size}",
              f"--mesh_tp={tp_size}",
              "--batch_size=4",
              "--mini_batch_size=4",
              "--num_generations=4",
              f"--max_prompt_length={prompt}",
              f"--max_response_length={response}",
              f"--env_max_steps={turns}",
              "--max_concurrency=16",
              "--sampler_is=none",
          ) + candidate_args:
            self.assertIn(argument, command)
          self.assertEqual(command.count("--sampler_is=none"), 1)
          with self.assertRaisesRegex(ValueError, "only backward-no-commit"):
            workload.command(run_stage="one-update")

    for production_name in ("frozenlake", "frozenlake-dp8-tp8"):
      with self.subTest(production_name=production_name):
        production_command = dp_workloads.get_workload(
            production_name
        ).command()
        self.assertNotIn("--sampler_is=none", production_command)

  def test_all_proxy_environments_pass_and_wrong_seams_reject(self):
    for name in (
        f"frozenlake-{recipe}-onehost-dp{dp_size}-tp{tp_size}"
        for recipe in ("p45", "m15")
        for dp_size, tp_size in ((4, 1), (2, 2), (1, 4))
    ):
      workload = dp_workloads.get_workload(name)
      environ = _environment(name)
      dp_workloads.validate_environment(
          workload, environ, require_reduction_admission=True
      )
      self.assertEqual(dp_workloads.requested_max_steps(workload, environ), 1)
      self.assertEqual(
          dp_workloads.expected_token_widths(workload, environ),
          (workload.max_prompt_length, workload.max_response_length),
      )
      dp_workloads.validate_frozenlake_max_concurrency(workload, 16, environ)
      for key, replacement in (
          ("CANON_PROFILE_FILE", "cluster/profiles/not-this.env"),
          ("CANON_MODEL_DIR_NAME", "not-this-model"),
          ("CANON_TP_SIZE", "8"),
          (
              "CANON_P66_P59_CHECK_VMA",
              "0" if workload.tp_size > 1 else "1",
          ),
          ("CANON_P33_RUN_STAGE", "one-update"),
      ):
        with self.subTest(name=name, key=key), self.assertRaises(ValueError):
          dp_workloads.validate_environment(
              workload,
              {**environ, key: replacement},
              require_reduction_admission=True,
          )

      if name == "frozenlake-p45-onehost-dp4-tp1":
        dp_workloads.validate_environment(
            workload,
            {**environ, "CANON_P78_SEGMENTED_ACTOR_LOGPS": "0"},
            require_reduction_admission=True,
        )
      else:
        with self.assertRaisesRegex(
            ValueError, "segmented actor logps require"
        ):
          dp_workloads.validate_environment(
              workload,
              {**environ, "CANON_P78_SEGMENTED_ACTOR_LOGPS": "1"},
              require_reduction_admission=True,
          )

  def test_p78_delivery_is_default_on_only_for_p45_dp4(self):
    runner = (
        ROOT
        / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts/"
        "run_frozenlake_dp2tp2_onehost.sh"
    ).read_text(encoding="utf-8")
    inner = (
        ROOT
        / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts/"
        "run_frozenlake_dp2tp2_inner.sh"
    ).read_text(encoding="utf-8")
    self.assertIn(
        '-e CANON_P78_SEGMENTED_ACTOR_LOGPS="$segmented_actor_logps"',
        runner,
    )
    self.assertIn(
        'V2_FL_P78_SEGMENTED_ACTOR_LOGPS:-$segmented_actor_logps_default',
        runner,
    )
    self.assertIn("P78 segmented actor logps admit only P45 DP4xTP1", runner)
    self.assertIn("profile changed P78 selector", inner)

  def test_each_matrix_workload_rejects_bent_topology_and_model(self):
    for name in (
        f"frozenlake-{recipe}-onehost-dp{dp_size}-tp{tp_size}"
        for recipe in ("p45", "m15")
        for dp_size, tp_size in ((4, 1), (2, 2), (1, 4))
    ):
      workload = dp_workloads.get_workload(name)
      for field, value in (
          ("dp_size", 8),
          ("tp_size", 8),
          ("model_dir_name", "qwen8b_tp8"),
          ("local_trajectories", workload.local_trajectories + 1),
          ("max_response_length", workload.max_response_length - 1),
      ):
        with self.subTest(name=name, field=field), self.assertRaises(
            ValueError
        ):
          dataclasses.replace(workload, **{field: value}).validate()

  def test_new_topology_profiles_resolve_exact_default_and_sorting_arms(self):
    for recipe in ("p45", "m15"):
      for dp_size, tp_size, model_dir, arm in (
          (4, 1, "qwen8b_tp1", ("stream", "1", "0")),
          (1, 4, "qwen8b", ("stream", "0", "0")),
          (1, 4, "qwen8b", ("stream", "0", "1")),
      ):
        keep_tape, reduce_once, length_sort = arm
        result = _source_matrix_profile(
            dp_size=dp_size,
            tp_size=tp_size,
            recipe=recipe,
            keep_tape=keep_tape,
            reduce_once=reduce_once,
            length_sort=length_sort,
        )
        expected_hbm = (
            "0.52" if recipe == "p45" else "0.66"
        ) if dp_size == 4 else "0.56"
        with self.subTest(recipe=recipe, dp=dp_size, tp=tp_size, arm=arm):
          self.assertEqual(result.returncode, 0, result.stderr)
          self.assertEqual(
              result.stdout.splitlines()[-1],
              (
                  f"frozenlake-{recipe}-onehost-dp{dp_size}-tp{tp_size}|"
                  f"{model_dir}|{dp_size}|{tp_size}|{keep_tape}|"
                  f"{reduce_once}|{length_sort}|"
                  f"{'0|0' if tp_size == 1 else '1|1'}|"
                  "0,2,1,3|0,2,1,3|"
                  f"{expected_hbm}"
              ),
          )
          self.assertIn("PROFILE_CONTRACT_PASS", result.stdout.splitlines())

  def test_dp1_tp4_profile_pins_the_physical_onehost_mesh_order(self):
    wrapper = (
        ROOT
        / "canon-zero-tim/cluster/profiles/"
        "qwen3-8b-dp1-tp4-frozenlake-onehost.env"
    ).read_text(encoding="utf-8")
    shared = (
        ROOT
        / "canon-zero-tim/cluster/profiles/"
        "_qwen3-8b-frozenlake-four-chip-onehost.env"
    ).read_text(encoding="utf-8")
    self.assertIn("_CANON_FL4_MESH_IDS=0,2,1,3", wrapper)
    self.assertIn(
        "qwen3-8b-dp1-tp4-frozenlake-onehost.env:1:4:qwen8b:1:0,2,1,3",
        shared,
    )
    self.assertNotIn(
        "qwen3-8b-dp1-tp4-frozenlake-onehost.env:1:4:qwen8b:1:0,1,2,3",
        shared,
    )

  def test_onehost_inner_asserts_the_created_training_mesh_order(self):
    inner = (
        ROOT
        / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts/"
        "run_frozenlake_dp2tp2_inner.sh"
    ).read_text(encoding="utf-8")
    self.assertIn("mesh_utils.create_device_mesh(", inner)
    self.assertIn("allow_split_physical_axes=True", inner)
    self.assertIn('os.environ["CANON_EXPECT_TRAIN_MESH_IDS"]', inner)
    self.assertIn("if actual_ids != expected_ids:", inner)
    self.assertIn("[V2.FL.ONEHOST] TRAIN_MESH_PASS", inner)

  def test_dp4_p45_hbm_cap_is_smallest_two_decimal_full_width_envelope(self):
    # R8 measured these aggregate bytes after removing the unused reference.
    total_hbm_bytes = 382.97 * 1024**3
    model_resident_bytes = 183.08 * 1024**3
    kv_bytes_per_token = 36 * 2 * 8 * 128 * 2
    full_width_kv_bytes = 16 * 6400 * kv_bytes_per_token
    self.assertLess(
        0.51 * total_hbm_bytes - model_resident_bytes,
        full_width_kv_bytes,
    )
    self.assertGreaterEqual(
        0.52 * total_hbm_bytes - model_resident_bytes,
        full_width_kv_bytes,
    )

  def test_new_topology_profiles_reject_crossed_seams(self):
    cases = (
        _source_matrix_profile(
            dp_size=1,
            tp_size=4,
            recipe="p45",
            keep_tape="stream",
            reduce_once="1",
            length_sort="0",
        ),
        _source_matrix_profile(
            dp_size=4,
            tp_size=1,
            recipe="p45",
            keep_tape="stream",
            reduce_once="1",
            length_sort="0",
            overrides={"CANON_P75_REPORT_ADJOINT_BUCKETS": "1"},
        ),
        _source_matrix_profile(
            dp_size=1,
            tp_size=4,
            recipe="m15",
            keep_tape="stream",
            reduce_once="0",
            length_sort="0",
            overrides={"CANON_P57_DATA_SPLIT": "selection"},
        ),
    )
    for result in cases:
      self.assertNotEqual(result.returncode, 0, result.stdout)

  def test_m15_candidate_pair_cannot_cross_into_p45(self):
    p45 = dp_workloads.get_workload("frozenlake-p45-onehost-dp2-tp2")
    with self.assertRaisesRegex(ValueError, "candidate/split"):
      dp_workloads.expected_token_widths(
          p45,
          {
              "CANON_P57_WORKLOAD_CANDIDATE": "m15",
              "CANON_P57_DATA_SPLIT": "main",
          },
      )

  def test_profile_accepts_registered_optimization_arms(self):
    script = f"""
set -euo pipefail
export CANON_PROFILE_FILE={PROFILE_REL}
export CANON_P57_WORKLOAD_CANDIDATE=m15
export CANON_P57_DATA_SPLIT=main
export CANON_P57_RUN_KIND=
export CANON_P57_TIM_ARM=
export CANON_P32_KEEP_TAPE=stream
export CANON_DP_REDUCE_ONCE=1
export CANON_P32_LENGTH_SORT=1
export CANON_P75_REPORT_ADJOINT_BUCKETS=0
export CANON_P76_CHUNK_DEPENDENCY_TICKET=0
export CANON_P77_CHUNK_BACKPRESSURE=0
source {PROFILE}
printf '%s\n' "$CANON_P32_WORKLOAD|$CANON_MODEL_DIR_NAME|$CANON_P66_P59_CHECK_VMA|$CANON_P33_RUN_STAGE|$CANON_P33_NO_COMMIT|${{CANON_DP_COLLECTIVE_REDUCE+x}}|$FL_VLLM_HBM_UTIL"
"""
    result = subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={},
    )
    self.assertEqual(
        result.stdout.strip(),
        "frozenlake-m15-onehost-dp2-tp2|qwen8b_tp2|1|backward-no-commit|1||0.37",
    )
    p45_script = script.replace(
        "export CANON_P57_WORKLOAD_CANDIDATE=m15",
        "export CANON_P57_WORKLOAD_CANDIDATE=",
    ).replace(
        "export CANON_P57_DATA_SPLIT=main",
        "export CANON_P57_DATA_SPLIT=",
    )
    p45 = subprocess.run(
        ["bash", "-c", p45_script],
        check=True,
        capture_output=True,
        text=True,
        env={},
    )
    self.assertEqual(
        p45.stdout.strip(),
        "frozenlake-p45-onehost-dp2-tp2|qwen8b_tp2|1|backward-no-commit|1||0.35",
    )
    invalid = script.replace(
        "export CANON_P32_KEEP_TAPE=stream",
        "export CANON_P32_KEEP_TAPE=0",
    )
    rejected = subprocess.run(
        ["bash", "-c", invalid],
        check=False,
        capture_output=True,
        text=True,
        env={},
    )
    self.assertNotEqual(rejected.returncode, 0)
    self.assertIn("optimization arm is not registered", rejected.stderr)

  def test_profile_admits_p76_only_with_p75_on_p45(self):
    script = f"""
set -euo pipefail
export CANON_PROFILE_FILE={PROFILE_REL}
export CANON_P57_WORKLOAD_CANDIDATE=
export CANON_P57_DATA_SPLIT=
export CANON_P57_RUN_KIND=
export CANON_P57_TIM_ARM=
export CANON_P32_KEEP_TAPE=0
export CANON_DP_REDUCE_ONCE=0
export CANON_P32_LENGTH_SORT=0
export CANON_P75_REPORT_ADJOINT_BUCKETS=1
export CANON_P76_CHUNK_DEPENDENCY_TICKET=1
export CANON_P77_CHUNK_BACKPRESSURE=0
source {PROFILE}
printf '%s\n' "$CANON_P32_WORKLOAD|$CANON_P75_REPORT_ADJOINT_BUCKETS|$CANON_P76_CHUNK_DEPENDENCY_TICKET|$CANON_P77_CHUNK_BACKPRESSURE"
"""
    accepted = subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={},
    )
    self.assertEqual(
        accepted.stdout.strip(),
        "frozenlake-p45-onehost-dp2-tp2|1|1|0",
    )
    for mutation in (
        script.replace(
            "export CANON_P75_REPORT_ADJOINT_BUCKETS=1",
            "export CANON_P75_REPORT_ADJOINT_BUCKETS=0",
        ),
        script.replace(
            "export CANON_P57_WORKLOAD_CANDIDATE=",
            "export CANON_P57_WORKLOAD_CANDIDATE=m15",
        ).replace(
            "export CANON_P57_DATA_SPLIT=",
            "export CANON_P57_DATA_SPLIT=main",
        ),
    ):
      rejected = subprocess.run(
          ["bash", "-c", mutation],
          check=False,
          capture_output=True,
          text=True,
          env={},
      )
      self.assertNotEqual(rejected.returncode, 0)

  def test_p75_p76_p77_delivery_is_exactly_p45_capacity_arms(self):
    runner = (
        ROOT
        / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts/"
        "run_frozenlake_dp2tp2_onehost.sh"
    ).read_text(encoding="utf-8")
    inner = (
        ROOT
        / "canon-zero-tim/tasks/v2-frozenlake-onehost/scripts/"
        "run_frozenlake_dp2tp2_inner.sh"
    ).read_text(encoding="utf-8")
    profile = PROFILE.read_text(encoding="utf-8")
    self.assertIn(
        '-e CANON_P75_REPORT_ADJOINT_BUCKETS="$report_adjoint_buckets"',
        runner,
    )
    self.assertIn(
        '-e CANON_P76_CHUNK_DEPENDENCY_TICKET="$chunk_dependency_ticket"',
        runner,
    )
    self.assertIn(
        '-e CANON_P77_CHUNK_BACKPRESSURE="$chunk_backpressure"', runner
    )
    self.assertIn(
        'if [ "${CANON_P75_REPORT_ADJOINT_BUCKETS:-}" != "$report_buckets" ]',
        inner,
    )
    self.assertIn(
        'if [ "${CANON_P76_CHUNK_DEPENDENCY_TICKET:-}" != "$chunk_ticket" ]',
        inner,
    )
    self.assertIn(
        'if [ "${CANON_P77_CHUNK_BACKPRESSURE:-}" != '
        '"$chunk_backpressure" ]',
        inner,
    )
    self.assertIn(
        "0:0:0:1:0:0|0:0:0:1:1:0|0:0:0:1:0:1)",
        profile,
    )
    self.assertIn(
        '"$CANON_P32_WORKLOAD" != "frozenlake-p45-onehost-dp2-tp2"',
        profile,
    )
    self.assertIn(
        'r0c) report_adjoint_buckets=1; chunk_dependency_ticket=1',
        runner,
    )
    self.assertIn(
        'r0c) keep_tape=0; reduce_once=0; length_sort=0; '
        'report_buckets=1; chunk_ticket=1',
        inner,
    )
    self.assertIn(
        'r0d) report_adjoint_buckets=1; chunk_dependency_ticket=0; '
        'chunk_backpressure=1',
        runner,
    )
    self.assertIn(
        'r0d) keep_tape=0; reduce_once=0; length_sort=0; '
        'report_buckets=1; chunk_ticket=0; chunk_backpressure=1',
        inner,
    )

  def test_profile_admits_p77_only_with_p75_on_p45(self):
    script = f"""
set -euo pipefail
export CANON_PROFILE_FILE={PROFILE_REL}
export CANON_P57_WORKLOAD_CANDIDATE=
export CANON_P57_DATA_SPLIT=
export CANON_P57_RUN_KIND=
export CANON_P57_TIM_ARM=
export CANON_P32_KEEP_TAPE=0
export CANON_DP_REDUCE_ONCE=0
export CANON_P32_LENGTH_SORT=0
export CANON_P75_REPORT_ADJOINT_BUCKETS=1
export CANON_P76_CHUNK_DEPENDENCY_TICKET=0
export CANON_P77_CHUNK_BACKPRESSURE=1
source {PROFILE}
printf '%s\n' "$CANON_P32_WORKLOAD|$CANON_P75_REPORT_ADJOINT_BUCKETS|$CANON_P76_CHUNK_DEPENDENCY_TICKET|$CANON_P77_CHUNK_BACKPRESSURE"
"""
    accepted = subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={},
    )
    self.assertEqual(
        accepted.stdout.strip(),
        "frozenlake-p45-onehost-dp2-tp2|1|0|1",
    )
    for mutation in (
        script.replace(
            "export CANON_P75_REPORT_ADJOINT_BUCKETS=1",
            "export CANON_P75_REPORT_ADJOINT_BUCKETS=0",
        ),
        script.replace(
            "export CANON_P76_CHUNK_DEPENDENCY_TICKET=0",
            "export CANON_P76_CHUNK_DEPENDENCY_TICKET=1",
        ),
        script.replace(
            "export CANON_P57_WORKLOAD_CANDIDATE=",
            "export CANON_P57_WORKLOAD_CANDIDATE=m15",
        ).replace(
            "export CANON_P57_DATA_SPLIT=",
            "export CANON_P57_DATA_SPLIT=main",
        ),
    ):
      rejected = subprocess.run(
          ["bash", "-c", mutation],
          check=False,
          capture_output=True,
          text=True,
          env={},
      )
      self.assertNotEqual(rejected.returncode, 0)


if __name__ == "__main__":
  unittest.main()
