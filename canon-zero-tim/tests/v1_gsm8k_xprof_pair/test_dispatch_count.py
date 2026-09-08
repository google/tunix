"""CPU contract for the per-update dispatch-count census."""

import importlib.util
import pathlib

SCRIPT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "tasks/v1-gsm8k-onehost-xprof-pair/scripts/census_dispatch_count.py"
)


def _module():
  spec = importlib.util.spec_from_file_location("census_dispatch_count", SCRIPT)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def test_families_classify_named_programs_before_glue():
  census = _module()
  assert census.family_of("jit_zt_tr_fwd_layer") == "fwd_layer"
  assert census.family_of("jit_zt_tr_dp_parallel_bwd_layer_27") == "bwd_layer"
  assert census.family_of("jit_zt_tr_fwd_logprob") == "rows"
  assert census.family_of("jit_zt_tr_bwd_logprob") == "rows"
  assert census.family_of("jit_rebuild") == "entry_cache"
  assert census.family_of("jit_zt_tr_grad_tree_add") == "grad_add"
  assert census.family_of("jit_group_pack") == "group_glue"
  assert census.family_of("jit_fused_chunk_metadata") == "group_glue"
  assert census.family_of("jit_convert_element_type") == "glue_eager"
  assert census.family_of("jit__multi_slice") == "other"


def test_chunk_passes_sum_the_captured_update_from_raw_log(tmp_path):
  census = _module()
  raw = tmp_path / "raw.log"
  lines = ["[P32.DP2] forward_group_issued group=%d/2 rows=(0, 1) n_real=(%d, %d)\n" % row
           for row in ((1, 100, 300), (2, 257, 12), (1, 600, 1), (2, 5, 5))]
  raw.write_text("".join(lines), encoding="utf-8")
  passes, per_group = census.chunk_passes_from_raw_log(raw, groups=2, bucket=256)
  # Only the last two groups (the captured update): ceil(600/256)=3, ceil(5/256)=1.
  assert per_group == [3, 1] and passes == 4


def test_chunk_passes_refuse_a_short_raw_log(tmp_path):
  census = _module()
  raw = tmp_path / "raw.log"
  raw.write_text("[P32.DP2] forward_group_issued group=1/2 rows=(0, 1) n_real=(1, 2)\n")
  try:
    census.chunk_passes_from_raw_log(raw, groups=2, bucket=256)
  except ValueError as exc:
    assert "forward_group_issued" in str(exc)
  else:
    raise AssertionError("expected ValueError")
