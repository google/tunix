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

from absl.testing import absltest
from tunix.dist_rl.common import otel_registry


class OtelRegistryTest(absltest.TestCase):

  def test_spec_defaults_and_validation(self):
    spec = otel_registry.InstrumentSpec(
        name="tunix.test.latency",
        kind="histogram",
        role="rollout",
        unit="s",
        description="Test latency.",
    )
    self.assertEqual(spec.effective_title, "tunix.test.latency")
    self.assertEqual(spec.effective_y_label, "s")
    self.assertEqual(
        spec.effective_aggregations,
        (("ALIGN_DELTA", "REDUCE_PERCENTILE_50"),),
    )

    with self.assertRaisesRegex(ValueError, "invalid for 'counter'"):
      otel_registry.InstrumentSpec(
          name="tunix.test.bad",
          kind="counter",
          role="rollout",
          unit="1",
          description="Bad.",
          aggregations=(("ALIGN_MEAN", "REDUCE_MEAN"),),
      )

    with self.assertRaisesRegex(ValueError, "high-cardinality"):
      otel_registry.InstrumentSpec(
          name="tunix.test.card",
          kind="gauge",
          role="rollout",
          unit="1",
          description="Bad label.",
          group_by=("worker_id",),
      )

  def test_register_is_idempotent_and_rejects_conflicts(self):
    spec = otel_registry.InstrumentSpec(
        name="tunix.test.idempotent",
        kind="counter",
        role="custom_role",
        unit="1",
        description="Count.",
    )
    self.addCleanup(otel_registry._REGISTRY.pop, spec.name, None)
    self.assertIs(otel_registry.register(spec), spec)
    self.assertEqual(otel_registry.register(spec), spec)
    with self.assertRaisesRegex(ValueError, "Conflicting"):
      otel_registry.register(
          otel_registry.InstrumentSpec(
              name="tunix.test.idempotent",
              kind="gauge",
              role="custom_role",
              unit="1",
              description="Different kind.",
          )
      )

  def test_specs_by_role_preserves_canonical_order_and_custom_roles(self):
    specs = (
        otel_registry.InstrumentSpec(
            name="tunix.custom.x",
            kind="gauge",
            role="custom_pool",
            unit="1",
            description="Custom.",
        ),
        otel_registry.InstrumentSpec(
            name="tunix.trainer.x",
            kind="gauge",
            role="trainer",
            unit="1",
            description="Trainer.",
        ),
        otel_registry.InstrumentSpec(
            name="tunix.rollout.x",
            kind="gauge",
            role="rollout",
            unit="1",
            description="Rollout.",
        ),
    )
    grouped = otel_registry.specs_by_role(specs)
    self.assertEqual(list(grouped), ["rollout", "trainer", "custom_pool"])
    self.assertEqual(otel_registry.role_title("custom_pool"), "Custom Pool")


if __name__ == "__main__":
  absltest.main()
