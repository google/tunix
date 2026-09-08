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

"""Preserve selector values, lazy reads and adapter error boundaries."""

import importlib
from pathlib import Path
from unittest import mock

import pytest

from tunix.rl import canonical_qwen3_adapter as adapter
from tunix.rl import canonical_training_config as config
from tunix.rl import dp_training


_OPTIONS = (
    ("CANON_P32_KEEP_TAPE", config.keep_tape_mode,
     {"": "", "0": "", "1": "batch", "stream": "stream"},
     "must be unset, 0, 1, or stream"),
    ("CANON_DP_REDUCE_ONCE", config.reduce_once_enabled,
     {"": False, "0": False, "1": True}, "must be unset, 0, or 1"),
    ("CANON_P59_RANK_PARALLEL_BACKWARD", config.rank_parallel_backward,
     {"": False, "0": False, "1": True}, "must be unset/0/1"),
)
_RAW = (None, "", "0", "1", "stream", "batch", "off", "true", "2", " 1", "1 ")


@pytest.mark.parametrize("flag,parse,values,message", _OPTIONS)
@pytest.mark.parametrize("raw", _RAW)
def test_selector_truth_table(monkeypatch, flag, parse, values, message, raw):
  if raw is None:
    monkeypatch.delenv(flag, raising=False)
  else:
    monkeypatch.setenv(flag, raw)
  value = "" if raw is None else raw
  if value in values:
    actual = parse()
    assert type(actual) is type(values[value])
    assert actual == values[value]
  else:
    with pytest.raises(ValueError) as caught:
      parse()
    assert type(caught.value) is ValueError
    assert str(caught.value) == f"{flag} {message}, got {value!r}"


@pytest.mark.parametrize("flag,parse,values,message", _OPTIONS)
def test_reads_only_its_option_and_never_caches(
    monkeypatch, flag, parse, values, message
):
  del message
  for other, _, _, _ in _OPTIONS:
    monkeypatch.setenv(other, "invalid-unused-option")
  # Reload must not validate unused options or capture an environment snapshot.
  importlib.reload(config)
  for raw in ("1", "0", "1", ""):
    monkeypatch.setenv(flag, raw)
    assert parse() == values[raw]
  monkeypatch.delenv(flag)
  assert parse() == values[""]


@pytest.mark.parametrize("flag,parse,values,message", _OPTIONS)
def test_callers_keep_their_existing_error_type(
    monkeypatch, flag, parse, values, message
):
  del values
  monkeypatch.setenv(flag, "bad")
  with pytest.raises(adapter.FunctionalMappingError) as caught:
    parse(error_type=adapter.FunctionalMappingError)
  assert type(caught.value) is adapter.FunctionalMappingError
  assert str(caught.value) == f"{flag} {message}, got 'bad'"


@pytest.mark.parametrize("raw", _RAW)
def test_existing_keep_tape_wrapper_contract(monkeypatch, raw):
  if raw is None:
    monkeypatch.delenv("CANON_P32_KEEP_TAPE", raising=False)
  else:
    monkeypatch.setenv("CANON_P32_KEEP_TAPE", raw)
  values = _OPTIONS[0][2]
  value = "" if raw is None else raw
  if value in values:
    assert adapter._p32_keep_tape_mode() == values[value]
    assert adapter._p32_keep_tape() is bool(values[value])
  else:
    with pytest.raises(adapter.FunctionalMappingError):
      adapter._p32_keep_tape_mode()


@pytest.mark.parametrize("raw", _RAW)
def test_existing_reduce_once_wrapper_contract(monkeypatch, raw):
  if raw is None:
    monkeypatch.delenv("CANON_DP_REDUCE_ONCE", raising=False)
  else:
    monkeypatch.setenv("CANON_DP_REDUCE_ONCE", raw)
  values = _OPTIONS[1][2]
  value = "" if raw is None else raw
  if value in values:
    assert dp_training.dp_reduce_once_mode() is values[value]
  else:
    with pytest.raises(ValueError) as caught:
      dp_training.dp_reduce_once_mode()
    assert type(caught.value) is ValueError


def test_real_reverse_keeps_rank_selector_as_first_boundary(monkeypatch):
  monkeypatch.setenv("CANON_P59_RANK_PARALLEL_BACKWARD", "bad")
  with mock.patch.object(adapter, "_p76_chunk_dependency_ticket_enabled") as next_gate:
    with pytest.raises(adapter.FunctionalMappingError) as caught:
      adapter.Qwen3EngineForwardAdapter._p32_reverse_group(
          None, None, None, None, None, None
      )
    next_gate.assert_not_called()
  assert str(caught.value) == (
      "CANON_P59_RANK_PARALLEL_BACKWARD must be unset/0/1, got 'bad'"
  )


@pytest.mark.parametrize("relative,count", (
    ("v1-gsm8k-onehost-xprof-pair/scripts/run_onehost_gsm8k_xprof_common.sh", 2),
    ("v2-frozenlake-onehost/scripts/run_frozenlake_dp2tp2_onehost.sh", 1),
))
def test_run_identity_includes_the_new_runtime_dependency(relative, count):
  repo = Path(__file__).resolve().parents[2]
  source = (repo / "canon-zero-tim/tasks" / relative).read_text()
  assert source.count('"$repo/tunix/rl/canonical_training_config.py"') == count
