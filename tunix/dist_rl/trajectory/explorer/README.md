<!-- TODO(timsemenov): Consolidate experimental trajectory documentation into a central docs location. -->
# Trajectory Explorer CLI (`trajectory-explore`)

`trajectory-explore` is a read-only diagnostic CLI for inspecting agentic RL rollouts and trajectories stored in a `TrajectoryStore`.

## Core Design Concepts

1. **Strict Read-Only Protocol Boundary (`TrajectoryReader`)**
   The CLI parses backend store flags (`--store`, `--root_dir`, `--run_id`) via `simple_parsing.subgroups`, builds the store config dict via `build_store_config()`, constructs the selected backend through `store.TrajectoryStore.from_config()`, and passes it to subcommands as a `store.TrajectoryReader`. Subcommands only ever receive a `TrajectoryReader`, guaranteeing that inspection tools cannot mutate or corrupt rollout data or couple to backend-specific storage internals.

2. **Dual Output Contract (Human & `--json`)**
   Every subcommand implements `execute(self, reader: store.TrajectoryReader, output_json: bool = False) -> int`.
   - When `output_json=False` (default), commands print formatted, human-readable terminal summaries.
   - When `output_json=True` (`--json`), commands emit machine-readable JSON to `stdout` while any CLI usage errors are isolated to `stderr` (exit code `2`), enabling clean composition with `jq` and automated evaluation scripts.

3. **Decoupled Testing Architecture**
   CLI flag parsing and store initialization are tested once in `run_explorer_test.py`. Individual subcommand tests (`commands/*_test.py`) do **not** touch the filesystem or CLI parser; instead, they test command logic directly against prefilled in-memory stores using the test harness in `commands/testing.py`.

4. **Decoupled Diagnostics Library (`diagnostics/`)**
  Statistics live in `diagnostics/stats.py` (`:diagnostics`), which has no CLI dependencies: `TrajectoryStats.from_trajectory()` summarizes one trajectory and `RunStats.from_trajectories()` aggregates a run into a `pandas.DataFrame` (`RunStats.df`, columns per `COLUMN_DTYPES`) whose columns are reduced into run totals. Unreported quantities are `None` (nulls in the DataFrame), never `0`. This holds for each token count on its own, so a trajectory that reports only completion tokens renders as `N completion`. Both classes offer `to_json_dict()` for the `--json` contract. Subcommands render these objects rather than computing their own figures, so notebooks and evaluation scripts can import the same numbers.

5. **Exit Codes**
   `0` on success, `1` when the requested data does not exist (e.g. `show` on an unknown trajectory ID), and `2` for a bad flag or flag combination. Errors go to `stderr`, never into the `--json` payload.

---

## Subcommands

| Command | What it prints |
|---|---|
| `ping` | Store connectivity and the number of trajectories. |
| `summary` | One row per trajectory (status, steps by source, tokens, tool calls, reward) followed by run totals: steps, tokens, status counts, mean reward and per-tool call counts. Under `--json`, the rows are the records of `RunStats.df`. |
| `show --trajectory_id=<id>` | A trajectory's header (status, reward, steps, tokens, subagents), every step with its message, reasoning, tool calls and observations (truncated to `--max_chars`, `0` for no truncation), and the tool-call sequence. |

---

## Usage

### Human-Readable Output
```bash
python3 -m tunix.dist_rl.trajectory.explorer.run_explorer \
  --store file \
  --root_dir /tmp/tx_smoke \
  --run_id run_95d297b1 \
  summary
```

```
trajectory  status     steps  agent  env  tokens        tool calls  reward
t_atif      SUCCEEDED      3      1    2  150 (100+50)           1  3.5
t_atif_2    SUCCEEDED      2      1    1  150 (100+50)           1  3.5

Trajectories: 2
Steps: 5 (agent 2, user 3, system 0, other 0)
Tokens: 300 (200+100) reported by 2/2
Statuses: SUCCEEDED x2 reported by 2/2
Mean reward: 3.5 reported by 2/2
Tool calls: search x2
```

### Machine-Readable JSON Output
```bash
python3 -m tunix.dist_rl.trajectory.explorer.run_explorer \
  --store file \
  --root_dir /tmp/tx_smoke \
  --run_id run_95d297b1 \
  --json show --trajectory_id traj_1001 | jq .stats.tool_call_names
```

---

## How to Add a New Subcommand

Adding a new subcommand requires **4 steps**. `commands/summary.py` and `commands/show.py` are complete examples; shared terminal helpers (missing-value marker, status name) live in `commands/formatting.py`, and `StepCounts`/`TokenCounts` render themselves via `str()`.

### Step 1: Implement `BaseCommand` in `commands/<name>.py`
Create a frozen dataclass inheriting from `base.BaseCommand`. Any fields declared on the dataclass automatically become subcommand-specific CLI flags via `simple_parsing`.

```python
@dataclasses.dataclass(frozen=True, kw_only=True)
class MyCommand(base.BaseCommand):
  """One-line description shown in --help."""

  some_flag: str = simple_parsing.field(help="Becomes --some_flag.")

  def execute(
      self, reader: store.TrajectoryReader, output_json: bool = False
  ) -> int:
    ...  # Query `reader`, print JSON or human-readable output.
    return 0
```

See `commands/show.py` for the reference implementation, including `--json` output and error handling.

### Step 2: Register in `commands/__init__.py`
Add your command class to `COMMAND_REGISTRY`:

```python
from tunix.dist_rl.trajectory.explorer.commands import show

COMMAND_REGISTRY: dict[str, type[base.BaseCommand]] = {
    "ping": ping.PingCommand,
    "show": show.ShowCommand,
}
```

### Step 3: Write Unit Tests in `commands/<name>_test.py`
Use `testing.create_store()` and `testing.execute_command()` to test both human-readable and JSON output modes against an in-memory store pre-populated with sample trajectories (`TRAJECTORY_1`, `TRAJECTORY_2`):

```python
from absl.testing import absltest
from tunix.dist_rl.trajectory.explorer.commands import show
from tunix.dist_rl.trajectory.explorer.commands import testing


class ShowTest(absltest.TestCase):

  def test_show_json_output(self) -> None:
    mem_store = testing.create_store(prefill=True)
    cmd = show.ShowCommand(trajectory_id="traj_1001")
    exit_code, out = testing.execute_command(cmd, mem_store, output_json=True)
    self.assertEqual(exit_code, 0)
    self.assertIn("traj_1001", out)
```

### Step 4: Run Tests
```bash
pytest tests/dist_rl/trajectory/explorer/
```
