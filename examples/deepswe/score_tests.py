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

r"""Scores a test log against a task's FAIL_TO_PASS / PASS_TO_PASS contract.

This script is copied verbatim into every generated Harbor task under
`tests/score_tests.py` and executed by `tests/test.sh` inside the task
container, so it must stay dependency free (standard library only) and run on
the system Python 3 of SWE-bench style images (3.7+; `tests/test.sh` prefers
`/usr/bin/python3` over the repository's conda interpreter for that reason).

Usage (inside the container):

  python3 /tests/score_tests.py --log /logs/verifier/test_output.log \
      --config /tests/config.json --test_exit_code 0 \
      --reward_file /logs/verifier/reward.txt \
      --results_file /logs/verifier/results.json

`config.json` carries `FAIL_TO_PASS`, `PASS_TO_PASS` and `reward_mode`
(`binary` or `partial`). The reward written to `reward_file` is a float in
[0, 1]; `results_file` receives a JSON breakdown that verifier audits can
inspect.
"""

import argparse
import json
import re
import sys
from typing import Any, Collection, Dict, Sequence, Set, Tuple

BINARY = 'binary'
PARTIAL = 'partial'

_PASS_STATUSES = frozenset({'PASSED', 'XFAIL', 'XPASS'})
_FAIL_STATUSES = frozenset({'FAILED', 'ERROR'})
_PYTEST_STATUSES = frozenset(
    {'PASSED', 'FAILED', 'ERROR', 'SKIPPED', 'XFAIL', 'XPASS'}
)
_PYTEST_STATUS_ALTERNATION = '|'.join(sorted(_PYTEST_STATUSES))

# pytest `-v` lines: `path::node STATUS`, optionally followed by a reason
# (`SKIPPED (no db)`) and / or a progress marker that is stripped beforehand.
_PYTEST_VERBOSE_LINE = re.compile(
    r'^(?P<id>\S.*?::.*?)\s+(?P<status>' + _PYTEST_STATUS_ALTERNATION + r')\b'
)
# pytest-xdist prefixes verbose lines with the worker id, e.g. `[gw0] PASSED`.
_XDIST_PREFIX = re.compile(r'^\[gw\d+\]\s+(?:\[\s*\d+%\]\s+)?')
# `test_name (module.Class) ... ok` lines emitted by Django / unittest runners.
_UNITTEST_LINE = re.compile(
    r'^(?P<name>\S.*?)\s+\.\.\.\s+(?P<status>ok|OK|FAIL|ERROR|skipped)\b'
)
_UNITTEST_PREFIX_LINE = re.compile(
    r'^(?P<status>FAIL|ERROR):\s+(?P<name>\w+ \([\w.]+\))\s*$'
)
# `test_func ok` / `test_func F` / `test_func E` lines emitted by SymPy's
# `bin/test --verbose`; the last test of a file also carries the file summary
# marker (`test_func ok    [OK]`). `f` marks an expected failure, `s` a skip.
_SYMPY_LINE = re.compile(
    r'^(?P<name>test_\w+)\s+(?:\.\.\.\s*)?'
    r'(?P<status>ok|OK|F|FAIL|E|ERROR|f|s)'
    r'(?:\s+\[(?:OK|FAIL)\])?\s*$'
)
_UNITTEST_STATUS_MAP = {
    'OK': 'PASSED',
    'FAIL': 'FAILED',
    'F': 'FAILED',
    'E': 'ERROR',
}
_SYMPY_STATUS_MAP = {
    **_UNITTEST_STATUS_MAP,
    'f': 'XFAIL',
    's': 'SKIPPED',
}
_ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*m|\r')
_PYTEST_PROGRESS_SUFFIX = re.compile(r'\s*\[\s*\d+%\]\s*$')
_DJANGO_ID_LINE = re.compile(r'^(?P<method>\w+) \((?P<cls>[\w.]+)\)$')


def strip_reason_suffix(test_id: str) -> str:
  """Drops a pytest ` - <reason>` suffix, ignoring ` - ` inside `[...]`."""
  depth = 0
  for index, char in enumerate(test_id):
    if char == '[':
      depth += 1
    elif char == ']':
      depth = max(0, depth - 1)
    elif depth == 0 and test_id.startswith(' - ', index):
      return test_id[:index]
  return test_id


def parse_statuses(log_text: str) -> Dict[str, str]:
  """Extracts `{test_id: STATUS}` from pytest / unittest / SymPy style output."""
  statuses: Dict[str, str] = {}
  pending_unittest_id = None
  cleaned_text = _ANSI_ESCAPE.sub('', log_text)
  for raw_line in cleaned_text.splitlines():
    line = _XDIST_PREFIX.sub('', raw_line.strip())
    line = _PYTEST_PROGRESS_SUFFIX.sub('', line).strip()
    if not line:
      continue
    first, _, rest = line.partition(' ')
    if first in _PYTEST_STATUSES and '::' in rest:
      statuses[strip_reason_suffix(rest).strip()] = first
      continue
    verbose_match = _PYTEST_VERBOSE_LINE.match(line)
    if verbose_match:
      statuses[verbose_match.group('id').strip()] = verbose_match.group(
          'status'
      )
      continue
    unittest_match = _UNITTEST_LINE.match(line) or _UNITTEST_PREFIX_LINE.match(
        line
    )
    if unittest_match:
      status = unittest_match.group('status').upper()
      statuses[unittest_match.group('name')] = _UNITTEST_STATUS_MAP.get(
          status, status
      )
      pending_unittest_id = None
      continue
    sympy_match = _SYMPY_LINE.match(line)
    if sympy_match:
      status = sympy_match.group('status')
      statuses[sympy_match.group('name')] = _SYMPY_STATUS_MAP.get(
          status, _SYMPY_STATUS_MAP.get(status.upper(), status.upper())
      )
      continue
    if line.endswith(' ...'):
      pending_unittest_id = line[: -len(' ...')].strip()
    elif pending_unittest_id and (line == 'ok' or line.startswith('ok ')):
      statuses[pending_unittest_id] = 'PASSED'
      pending_unittest_id = None
  return statuses


def _is_ascii(text: str) -> bool:
  """Returns True when `text` only contains ASCII characters."""
  try:
    text.encode('ascii')
  except UnicodeEncodeError:
    return False
  return True


def _candidate_aliases(candidate: str) -> Tuple[str, ...]:
  """Returns normalized forms of a parsed test id for cross-runner matching."""
  aliases = [candidate]
  if '::' in candidate:
    file_part, _, node_rest = candidate.partition('::')
    dot_joined = node_rest.replace('::', '.')
    aliases.append(dot_joined)
    aliases.append(f'{file_part}::{dot_joined}')
  django_match = _DJANGO_ID_LINE.match(candidate)
  if django_match:
    method = django_match.group('method')
    cls = django_match.group('cls')
    if cls.endswith(f'.{method}'):
      cls = cls[: -len(method) - 1]
      aliases.append(f'{method} ({cls})')
    aliases.append(f'{cls}.{method}')
  return tuple(aliases)


class TestIdIndex:
  """Lookup structure over parsed test ids and their runner aliases."""

  def __init__(self, test_ids: Collection[str]):
    self._aliases: Set[str] = set()
    for test_id in test_ids:
      self._aliases.update(_candidate_aliases(test_id))
    self._base_names = {alias.split('[', 1)[0] for alias in self._aliases}

  def matches(self, expected: str) -> bool:
    """Returns True if `expected` corresponds to some indexed test id."""
    if expected in self._aliases:
      return True
    clean_expected = _ANSI_ESCAPE.sub('', expected).strip()
    expected_id = strip_reason_suffix(clean_expected)
    if clean_expected in self._aliases or expected_id in self._aliases:
      return True
    if '[' in expected_id and not expected_id.endswith(']'):
      return any(alias.startswith(expected_id) for alias in self._aliases)
    if not _is_ascii(expected_id):
      return expected_id.split('[', 1)[0] in self._base_names
    return False


def test_matches(expected: str, candidates: Collection[str]) -> bool:
  """Returns True if `expected` is in `candidates`, tolerating runner aliases."""
  return TestIdIndex(candidates).matches(expected)


def score(
    statuses: Dict[str, str],
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
    reward_mode: str,
    test_exit_code: int,
    strict_pass_to_pass: bool = False,
) -> Dict[str, Any]:
  """Computes the reward and a JSON serializable breakdown."""
  passed = TestIdIndex(
      [test for test, status in statuses.items() if status in _PASS_STATUSES]
  )
  failed = TestIdIndex(
      [test for test, status in statuses.items() if status in _FAIL_STATUSES]
  )

  f2p_passed = [test for test in fail_to_pass if passed.matches(test)]
  f2p_missing = [test for test in fail_to_pass if not passed.matches(test)]
  p2p_passed = [test for test in pass_to_pass if passed.matches(test)]
  p2p_failed = [test for test in pass_to_pass if failed.matches(test)]
  p2p_missing = [test for test in pass_to_pass if not passed.matches(test)]
  require_explicit_p2p = strict_pass_to_pass or not fail_to_pass
  p2p_regressions = p2p_missing if require_explicit_p2p else p2p_failed

  if fail_to_pass:
    f2p_fraction = len(f2p_passed) / len(fail_to_pass)
  elif pass_to_pass:
    f2p_fraction = 1.0
  else:
    f2p_fraction = 1.0 if test_exit_code == 0 else 0.0
  p2p_fraction = (
      1.0 - len(p2p_regressions) / len(pass_to_pass) if pass_to_pass else 1.0
  )

  notes = []
  if (fail_to_pass or pass_to_pass) and not statuses:
    notes.append('no test statuses could be parsed from the log')
  if reward_mode == PARTIAL:
    reward = f2p_fraction * p2p_fraction
  else:
    reward = 1.0 if f2p_fraction == 1.0 and not p2p_regressions else 0.0
  reward = min(1.0, max(0.0, reward))
  return {
      'reward': reward,
      'reward_mode': reward_mode,
      'strict_pass_to_pass': strict_pass_to_pass,
      'test_exit_code': test_exit_code,
      'parsed_status_count': len(statuses),
      'fail_to_pass': {
          'total': len(fail_to_pass),
          'passed': f2p_passed,
          'missing_or_failed': f2p_missing,
      },
      'pass_to_pass': {
          'total': len(pass_to_pass),
          'passed': p2p_passed,
          'failed': p2p_failed,
          'missing_or_failed': p2p_missing,
      },
      'notes': notes,
  }


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
  """Parses command line arguments (argparse keeps this script stdlib-only)."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--log', required=True, help='Captured test output.')
  parser.add_argument('--config', required=True, help='tests/config.json.')
  parser.add_argument('--test_exit_code', type=int, default=0)
  parser.add_argument('--reward_file', required=True)
  parser.add_argument('--results_file', default='')
  return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
  """Scores the log and writes the reward; returns the process exit code."""
  args = _parse_args(argv)
  with open(args.config, 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)
  try:
    with open(args.log, 'r', encoding='utf-8', errors='replace') as log_file:
      log_text = log_file.read()
  except OSError as error:
    print('score_tests: cannot read log: %s' % error, file=sys.stderr)
    log_text = ''

  results = score(
      statuses=parse_statuses(log_text),
      fail_to_pass=list(config.get('FAIL_TO_PASS', [])),
      pass_to_pass=list(config.get('PASS_TO_PASS', [])),
      reward_mode=config.get('reward_mode', BINARY),
      test_exit_code=args.test_exit_code,
      strict_pass_to_pass=bool(config.get('strict_pass_to_pass', False)),
  )
  with open(args.reward_file, 'w', encoding='utf-8') as reward_file:
    reward_file.write('%s\n' % results['reward'])
  if args.results_file:
    with open(args.results_file, 'w', encoding='utf-8') as results_file:
      json.dump(results, results_file, indent=2, sort_keys=True)
  print(
      'score_tests: reward=%s fail_to_pass=%d/%d pass_to_pass_failed=%d'
      % (
          results['reward'],
          len(results['fail_to_pass']['passed']),
          results['fail_to_pass']['total'],
          len(results['pass_to_pass']['failed']),
      )
  )
  return 0 if results['reward'] >= 1.0 else 1


if __name__ == '__main__':
  sys.exit(main(sys.argv[1:]))
