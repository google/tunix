# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for distributed runtime main entry point."""

from unittest import mock
from absl.testing import absltest
from tunix.experimental.distributed.runtime import main as distributed_main


def dummy_process_main(argv, context):
  pass


class MainTest(absltest.TestCase):

  def test_import_symbol_valid(self):
    fn = distributed_main.import_symbol(
        "tunix.experimental.distributed.runtime.main.split_process_argv"
    )
    self.assertIs(fn, distributed_main.split_process_argv)

  def test_import_symbol_invalid_no_dot(self):
    with self.assertRaises(ValueError):
      distributed_main.import_symbol("nodotsymbol")

  def test_import_symbol_module_not_found(self):
    with self.assertRaises(ModuleNotFoundError):
      distributed_main.import_symbol(
          "tunix.experimental.distributed.nonexistent.module.symbol"
      )

  def test_import_symbol_attribute_error(self):
    with self.assertRaises(AttributeError):
      distributed_main.import_symbol(
          "tunix.experimental.distributed.runtime.main.NonexistentSymbol"
      )

  def test_split_process_argv_no_process_flag(self):
    argv = ["--foo=bar", "--baz=123"]
    slices = distributed_main.split_process_argv(argv)
    self.assertEqual(slices, [["--foo=bar", "--baz=123"]])

  def test_split_process_argv_multiple_processes(self):
    argv = [
        "--process",
        "--process_main=foo.main",
        "--port=123",
        "--process",
        "--process_main=bar.main",
        "--port=456",
    ]
    slices = distributed_main.split_process_argv(argv)
    self.assertEqual(
        slices,
        [
            ["--process_main=foo.main", "--port=123"],
            ["--process_main=bar.main", "--port=456"],
        ],
    )

  def test_prepare_process_success(self):
    argv = [
        f"--process_main={__name__}.dummy_process_main",
        "--discovery_id=worker-0",
        "--discovery_port=8080",
        "--discovery_addrs=leader:8080",
        "--custom_flag=hello",
    ]
    prepared = distributed_main.prepare_process(argv)
    self.assertEqual(prepared.main_fn.__name__, dummy_process_main.__name__)
    self.assertEqual(prepared.context_args.discovery_id, "worker-0")
    self.assertEqual(prepared.context_args.discovery_port, 8080)
    self.assertEqual(prepared.context_args.discovery_addrs, "leader:8080")
    self.assertEqual(prepared.argv, ["--custom_flag=hello"])

  def test_prepare_process_invalid_main(self):
    argv = [f"--process_main={__name__}.nonexistent_fn"]
    with self.assertRaises(ValueError):
      distributed_main.prepare_process(argv)

  @mock.patch.object(distributed_main, "import_symbol")
  def test_main_single_process(self, mock_import_symbol):
    mock_executor = mock.MagicMock()
    mock_import_symbol.side_effect = lambda fqn: (
        mock_executor if "Executor" in fqn else dummy_process_main
    )

    argv = [
        "prog",
        "--process_executor=tunix.experimental.distributed.runtime.executor.LocalExecutor",
        f"--process_main={__name__}.dummy_process_main",
        "--custom_arg=abc",
    ]
    distributed_main.main(argv)
    mock_executor.return_value.run.assert_called_once()

  @mock.patch.object(distributed_main, "import_symbol")
  def test_main_multi_process(self, mock_import_symbol):
    mock_executor = mock.MagicMock()
    mock_import_symbol.side_effect = lambda fqn: (
        mock_executor if "Executor" in fqn else dummy_process_main
    )

    argv = [
        "prog",
        "--process_executor=tunix.experimental.distributed.runtime.executor.LocalExecutor",
        "--process",
        f"--process_main={__name__}.dummy_process_main",
        "--p=1",
        "--process",
        f"--process_main={__name__}.dummy_process_main",
        "--p=2",
    ]
    distributed_main.main(argv)
    self.assertEqual(mock_executor.return_value.run.call_count, 2)

  def _run_single(
      self, run_side_effect: BaseException, *extra_main_args: str
  ) -> None:
    mock_executor = mock.MagicMock()
    mock_executor.return_value.run.side_effect = run_side_effect
    with mock.patch.object(
        distributed_main,
        "import_symbol",
        side_effect=lambda fqn: (
            mock_executor if "Executor" in fqn else dummy_process_main
        ),
    ):
      distributed_main.main([
          "prog",
          *extra_main_args,
          f"--process_main={__name__}.dummy_process_main",
      ])

  def test_exit_on_failure_hard_exits_on_exception(self):
    with mock.patch.object(
        distributed_main, "_hard_exit", side_effect=_HardExit
    ) as hard_exit:
      with self.assertRaises(_HardExit):
        self._run_single(RuntimeError("boom"), "--exit_on_failure")
    hard_exit.assert_called_once_with(1)

  def test_exit_on_failure_propagates_nonzero_system_exit_code(self):
    with mock.patch.object(
        distributed_main, "_hard_exit", side_effect=_HardExit
    ) as hard_exit:
      with self.assertRaises(_HardExit):
        self._run_single(SystemExit(143), "--exit_on_failure")
    hard_exit.assert_called_once_with(143)

  def test_exit_on_failure_reraises_clean_system_exit(self):
    with mock.patch.object(distributed_main, "_hard_exit") as hard_exit:
      with self.assertRaises(SystemExit):
        self._run_single(SystemExit(0), "--exit_on_failure")
    hard_exit.assert_not_called()

  def test_without_exit_on_failure_exception_propagates(self):
    with mock.patch.object(distributed_main, "_hard_exit") as hard_exit:
      with self.assertRaisesRegex(RuntimeError, "boom"):
        self._run_single(RuntimeError("boom"))
    hard_exit.assert_not_called()


class _HardExit(Exception):
  """Stands in for os._exit so the test process keeps running."""


if __name__ == "__main__":
  absltest.main()
