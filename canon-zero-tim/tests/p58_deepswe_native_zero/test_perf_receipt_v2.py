"""tasks/deepswe_4b_perf perf receipt v2: runner patch 40 wiring and the accounting script."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "canon-zero-tim"
PATCH = PKG / "patches/tpu_inference/40-tpu-runner-perf-receipt-v2.patch"
SCRIPT = PKG / "tasks/p58-deepswe-native-zero-comparison/scripts/account_engine_steps.py"

V1 = ("[ENGINE_STEP] n=1 t=20.447 path=std reqs=8 prefill_reqs=1 sched_tok=512 pad_tok=2048 pad_reqs=8 "
      "pad_per_rank=256 per_rank_reqs=1 per_rank_tok=256 per_rank_prefill=1 mixed=0 gap_ms=1.00 exec_ms=8.00 "
      "sample_ms=100.00 sync_ms=1.00 cd_steps=0 cd_max=0 cd_eos=0 gen_tok=8")
V2 = ("[ENGINE_STEP] n=%d t=30.0 path=std reqs=100 prefill_reqs=8 sched_tok=2000 pad_tok=2048 pad_reqs=128 "
      "pad_per_rank=256 per_rank_reqs=12/13 per_rank_tok=250/250 per_rank_prefill=1/1 mixed=0 gap_ms=8.00 "
      "exec_ms=9.00 sample_ms=110.00 sync_ms=1.00 cd_steps=0 cd_max=0 cd_eos=0 gen_tok=100 fill=0.977 "
      "kv_tok=160000 synced=%d dev_model_ms=%.2f launch_ms=run_model:3.20/select:0.10/compute_logits:0.40/split:0.05/sample:0.80/logprob:0.50")


def _load():
  spec = importlib.util.spec_from_file_location("account_engine_steps", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


class PerfReceiptV2Test(unittest.TestCase):

  def test_patch_is_registered_after_39_and_adds_the_fields(self):
    install = (PKG / "install.sh").read_text()
    self.assertLess(install.index("39-tpu-runner-perf3-engine-step-log.patch"),
                    install.index("40-tpu-runner-perf-receipt-v2.patch"))
    patch = PATCH.read_text()
    for marker in ("CANON_ENGINE_STEP_LOG_SYNC_EVERY", "def _engine_step_launch(",
                   "def _engine_step_sync_sample(", "jax.block_until_ready(value)",
                   'fill=%.3f kv_tok=%d synced=%d dev_model_ms=%.2f launch_ms=%s',
                   '_engine_step_launch(self, "run_model", _perf3_t_run_model)',
                   '_engine_step_launch(self, "sample", _perf3_t_sample)',
                   '_engine_step_launch(self, "logprob", _perf3_t_logprob)',
                   "num_computed_tokens_cpu"):
      self.assertIn(marker, patch, marker)
    # the sync never runs on the continue-decode or empty paths
    self.assertIn('if _perf3_rec["path"] == "std":', patch)
    flags = (PKG / "FLAGS.md").read_text()
    self.assertIn("| `CANON_ENGINE_STEP_LOG_SYNC_EVERY` |", flags)
    self.assertIn("\nCANON_ENGINE_STEP_LOG_SYNC_EVERY\n", flags)
    self.assertIn("export CANON_ENGINE_STEP_LOG_SYNC_EVERY=200\n",
                  (PKG / "cluster/profiles/qwen3-4b-dp8-tp8-deepswe-tim.env").read_text())
    self.assertIn("export CANON_ENGINE_STEP_LOG_SYNC_EVERY=50\n",
                  (PKG / "cluster/profiles/qwen3-4b-dp1-tp4-deepswe-zero.env").read_text())

  def test_accounting_reads_v1_and_v2_receipts(self):
    module = _load()
    lines = [V1, V2 % (2, 0, 0.0), V2 % (3, 1, 70.5), V2 % (4, 0, 0.0),
             "2026-09-15T22:20:10Z [PERF] stage=rollout_generate seconds=38.741 rows=1",
             "[PERF] step=0 stage=rescore_b seconds=19.854 rows=4", "noise line"]
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "raw.log"
      path.write_text("\n".join(lines) + "\n")
      rows, perf = [], []
      for line in path.read_text().splitlines():
        row = module.parse_engine_line(line)
        if row:
          rows.append(row)
        stage = module.parse_perf_line(line)
        if stage:
          perf.append(stage)
      self.assertEqual(len(rows), 4)
      self.assertEqual(rows[0].get("fill"), None)
      self.assertEqual(rows[1]["fill"], 0.977)
      self.assertEqual(rows[1]["launch"]["run_model"], 3.2)
      report = module.account(rows, perf, [32, 256, 1024, 4096])
    self.assertIn("receipts: 4; paths: std=4", report)
    self.assertIn("| std | 1025-4096 | 3 |", report)
    self.assertIn("| std | 257-1024 | 1 |", report)
    self.assertIn("| 1025-4096 | 1 | 70.5 | 9.0 | 110.0 | 8.0 |", report)
    self.assertIn("| run_model | 3 | 3.20 | 3.20 |", report)
    self.assertIn("| rollout_generate | 1 | 38.7 | 38.74 |", report)
    self.assertIn("| rescore_b | 1 | 19.9 | 19.85 |", report)


if __name__ == "__main__":
  unittest.main()
