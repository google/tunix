#!/usr/bin/env python3
"""CPU oracle for the P58 fixed-16K compact-filter loss contract."""

from __future__ import annotations

import ast
from pathlib import Path
import types
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from tunix.rl import algo_core
from tunix.rl import common


ROOT = Path(__file__).resolve().parents[3]
DEEPSWE_SCRIPT = ROOT / "examples/deepswe/train_deepswe_nb.py"
ADAPTER = ROOT / "tunix/rl/canonical_qwen3_adapter.py"
FIXED_NORM = 16_384
RAW_ROWS = 128
MICROBATCH_ROWS = 16


class P58LossContractTest(unittest.TestCase):

  def test_fixed_norm_excludes_compact_filtered_rows(self):
    losses = jnp.asarray([[1.0, 2.0], [20.0, 30.0]])
    mask = jnp.asarray([[1.0, 1.0], [0.0, 0.0]])
    metric = common.aggregate_loss(
        losses,
        mask,
        "sequence-mean-token-scale",
        norm=FIXED_NORM,
    )
    self.assertEqual(float(metric.denominator), 1.0)
    self.assertAlmostEqual(float(metric.compute()), 3.0 / FIXED_NORM)
    self.assertAlmostEqual(
        float(
            common.reduced_loss_agg(
                losses,
                mask,
                "sequence-mean-token-scale",
                norm=FIXED_NORM,
            )
        ),
        3.0 / FIXED_NORM,
    )

  def test_all_filtered_is_finite_zero_with_zero_denominator(self):
    losses = jnp.ones((RAW_ROWS, 4), dtype=jnp.float32)
    mask = jnp.zeros_like(losses)
    metric = common.aggregate_loss(
        losses,
        mask,
        "sequence-mean-token-scale",
        norm=FIXED_NORM,
    )
    self.assertEqual(float(metric.denominator), 0.0)
    self.assertEqual(float(metric.compute()), 0.0)

  def test_empty_completion_admission_is_deepswe_scoped(self):
    tree = ast.parse(ADAPTER.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    group_spec = functions["_p32_group_spec"]
    keyword_names = [arg.arg for arg in group_spec.args.kwonlyargs]
    allow_index = keyword_names.index("allow_empty_completion")
    allow_default = group_spec.args.kw_defaults[allow_index]
    self.assertIsInstance(allow_default, ast.Constant)
    self.assertIs(allow_default.value, False)

    segmented = functions["segmented_dp_grpo_value_and_grad"]
    calls = [
        node
        for node in ast.walk(segmented)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_p32_group_spec"
    ]
    self.assertEqual(len(calls), 1)
    keywords = {item.arg: item.value for item in calls[0].keywords}
    allow_value = keywords.get("allow_empty_completion")
    self.assertIsInstance(allow_value, ast.Name)
    self.assertEqual(allow_value.id, "p34")

  def test_eight_unequal_effective_microbatches_match_full_gradient(self):
    coefficients = jnp.arange(
        1, RAW_ROWS * 4 + 1, dtype=jnp.float32
    ).reshape(RAW_ROWS, 4)
    effective_per_microbatch = (1, 15, 3, 13, 5, 11, 7, 9)
    mask_parts = []
    for effective in effective_per_microbatch:
      part = jnp.zeros((MICROBATCH_ROWS, 4), dtype=jnp.float32)
      part = part.at[:effective, :].set(1.0)
      mask_parts.append(part)
    mask = jnp.concatenate(mask_parts, axis=0)

    def full_loss(theta):
      return common.aggregate_loss(
          theta * coefficients,
          mask,
          "sequence-mean-token-scale",
          norm=FIXED_NORM,
      ).compute()

    full_gradient = jax.grad(full_loss)(jnp.asarray(1.0, jnp.float32))
    numerator_gradients = []
    denominators = []
    local_mean_gradients = []
    for index in range(8):
      start = index * MICROBATCH_ROWS
      stop = start + MICROBATCH_ROWS
      coeff_part = coefficients[start:stop]
      mask_part = mask[start:stop]

      def local_metric(theta):
        return common.aggregate_loss(
            theta * coeff_part,
            mask_part,
            "sequence-mean-token-scale",
            norm=FIXED_NORM,
        )

      numerator_gradients.append(
          jax.grad(lambda theta: local_metric(theta).unreduced_sum)(
              jnp.asarray(1.0, jnp.float32)
          )
      )
      denominators.append(local_metric(jnp.asarray(1.0)).denominator)
      local_mean_gradients.append(
          jax.grad(lambda theta: local_metric(theta).compute())(
              jnp.asarray(1.0, jnp.float32)
          )
      )

    accumulated = sum(numerator_gradients) / sum(denominators)
    naive_mean_of_means = sum(local_mean_gradients) / len(local_mean_gradients)
    np.testing.assert_allclose(accumulated, full_gradient, rtol=1e-6, atol=1e-6)
    self.assertGreater(
        abs(float(naive_mean_of_means - full_gradient)), 1e-4
    )

  def test_explicit_norm_rejects_compiled_width_drift(self):
    config = types.SimpleNamespace(
        loss_agg_mode="sequence-mean-token-scale",
        loss_scale_factor=FIXED_NORM,
    )
    matching = jnp.ones((2, FIXED_NORM), dtype=jnp.float32)
    self.assertEqual(
        algo_core._loss_aggregation_kwargs(config, matching),
        {"norm": FIXED_NORM},
    )
    with self.assertRaisesRegex(ValueError, "compiled response width"):
      algo_core._loss_aggregation_kwargs(config, matching[:, :-1])

  def test_overlong_filter_cli_is_not_type_bool(self):
    tree = ast.parse(DEEPSWE_SCRIPT.read_text(encoding="utf-8"))
    matching_calls = []
    for node in ast.walk(tree):
      if not isinstance(node, ast.Call):
        continue
      if not (
          isinstance(node.func, ast.Attribute)
          and node.func.attr == "add_argument"
      ):
        continue
      if not node.args or not isinstance(node.args[0], ast.Constant):
        continue
      if node.args[0].value == "--overlong_filter":
        matching_calls.append(node)
    self.assertEqual(len(matching_calls), 1)
    keywords = {item.arg: item.value for item in matching_calls[0].keywords}
    self.assertNotIn("type", keywords)
    action = keywords.get("action")
    self.assertIsInstance(action, ast.Attribute)
    self.assertEqual(action.attr, "BooleanOptionalAction")


if __name__ == "__main__":
  unittest.main()
