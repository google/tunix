<!-- TODO(timsemenov): Consolidate experimental trajectory documentation into a central docs location. -->
# Trajectory Explorer CLI (`trajectory-explore`)

`trajectory-explore` is a read-only diagnostic CLI for inspecting agentic RL rollouts and trajectories stored in a `TrajectoryStore`.

## Scope

- **Tunix entry point**: run as a program (`trajectory-explore`), `run_explorer.py` passes `TunixDiagnostics` (`diagnostics/tunix_diagnostics.py`), the statistics specific to Tunix RL rollouts, to the subcommands.
- **Schema-agnostic CLI and library**: the subcommands (`commands/`), `run_explorer.launch_cli()` and the rest of `diagnostics/` (`stats.py`, `schema_diagnostics.py`, `formatting.py`) depend on no particular trajectory schema: statistics specific to one schema come from the schema diagnostics they are given (see *Schema Diagnostics* below). The library has no CLI dependencies. They are planned to move into a standalone trajectory library, leaving `trajectory-explore` in Tunix as a thin entry point.

## Core Design Concepts

1. **Strict Read-Only Protocol Boundary (`TrajectoryReader`)**
   The CLI parses backend store flags (`--store`, `--root_dir`, `--run_id`) via `simple_parsing.subgroups`, builds the store config dict via `build_store_config()`, constructs the selected backend through `store.TrajectoryStore.from_config()`, and passes it to subcommands as a `store.TrajectoryReader`. Subcommands only ever receive a `TrajectoryReader`, guaranteeing that inspection tools cannot mutate or corrupt rollout data or couple to backend-specific storage internals.

2. **Dual Output Contract (Human & `--json`)**
   Every subcommand implements `execute(self, reader: store.TrajectoryReader, output_json: bool = False, *, diagnostics: Sequence[SchemaDiagnostics] = ()) -> int`, where `diagnostics` are the schema diagnostics the entry point passes to `launch_cli(diagnostics=...)`.
   - When `output_json=False` (default), commands print formatted, human-readable terminal summaries.
   - When `output_json=True` (`--json`), commands emit machine-readable JSON to `stdout` while any CLI usage errors are isolated to `stderr` (exit code `2`), enabling clean composition with `jq` and automated evaluation scripts.

3. **Decoupled Testing Architecture**
   CLI flag parsing and store initialization are tested once in `run_explorer_test.py`. Individual subcommand tests (`commands/*_test.py`) do **not** touch the filesystem or CLI parser; instead, they test command logic directly against prefilled in-memory stores using the test harness in `commands/testing.py`.

4. **Decoupled Diagnostics Library (`diagnostics/`)**
  Statistics live in `diagnostics/stats.py` (`:diagnostics`), which has no CLI dependencies: `TrajectoryStats.from_trajectory()` summarizes one trajectory and `RunStats.from_trajectories()` aggregates a run into a `pandas.DataFrame` (`RunStats.df`, columns per `COLUMN_DTYPES`) whose columns are reduced into run totals. Unreported quantities are `None` (nulls in the DataFrame), never `0`. This holds for each token count on its own, so a trajectory that reports only completion tokens renders as `N completion`. Both classes offer `to_json_dict()` for the `--json` contract. Subcommands render these objects rather than computing their own figures, so notebooks and evaluation scripts can import the same numbers. Statistics that only one trajectory schema reports come from schema diagnostics (see *Schema Diagnostics* below).

5. **Exit Codes**
   `0` on success, `1` when the requested data does not exist (e.g. `show` on an unknown trajectory ID), and `2` for a bad flag or flag combination. Errors go to `stderr`, never into the `--json` payload.

---

## Subcommands

| Command | What it prints |
|---|---|
| `ping` | Store connectivity and the number of trajectories. |
| `summary [--limit=<N>] [--show_trajectory_table]` | Run totals: steps, tokens, status counts, how the `SUCCEEDED` trajectories scored, mean reward, the totals of each schema diagnostics that applies, and per-tool call counts. `--show_trajectory_table` also prints one row per trajectory above them (status, steps by source, tokens, tool calls, reward, then the columns of each schema diagnostics that applies). Optional `--limit` bounds trajectory loading and table size. Under `--json`, the rows are always included, as the records of `RunStats.df`. |
| `show --trajectory_id=<id> [<id> ...]` | Each trajectory's header (status, reward, steps, tokens, subagents, then one line per value of each schema diagnostics that applies), every step with its message, reasoning, tool calls and observations (truncated to `--max_chars`, `0` for no truncation), and the tool-call sequence. Accepts space- or comma-separated IDs; multiple trajectories are separated by an `====` rule (`--json` emits a list when 2+ IDs are given). |

---

## Usage

### Human-Readable Output
```bash
python3 -m tunix.dist_rl.trajectory.explorer.run_explorer \
  --store file \
  --root_dir /tmp/tx_smoke \
  --run_id run_95d297b1 \
  summary --show_trajectory_table
```

```
trajectory  status     steps  agent  env  tokens        tool calls  reward
t_atif      SUCCEEDED      3      1    2  150 (100+50)           1  3.5
t_atif_2    SUCCEEDED      2      1    1  150 (100+50)           1  3.5

Trajectories: 2
Steps: 5 (agent 2, user 3, system 0, other 0)
Tokens: 300 (200+100) reported by 2/2
Statuses: SUCCEEDED x2 reported by 2/2
Outcome: 2 solved (r>0), 0 unsolved (r<=0), 0 unscored of 2 SUCCEEDED
Mean reward: 3.5 reported by 2/2
Tool calls: search x2
```

Without `--show_trajectory_table`, `summary` prints only the totals, from `Trajectories:` on. `Outcome:` splits the trajectories whose status is `SUCCEEDED` (`stats.SUCCEEDED_STATUS`) by reward into solved (`> 0`), unsolved (`<= 0`) and unscored (no reward); trajectories with any other status are not counted, and the line is omitted when no trajectory succeeded. Schema diagnostics that apply to the run (see *Schema Diagnostics* below) add their columns after `reward` and their totals after `Mean reward:`.

### Machine-Readable JSON Output
```bash
python3 -m tunix.dist_rl.trajectory.explorer.run_explorer \
  --store file \
  --root_dir /tmp/tx_smoke \
  --run_id run_95d297b1 \
  --json show --trajectory_id traj_1001 | jq .stats.tool_call_names
```

---

## Schema Diagnostics

`diagnostics/stats.py` only computes what every ATIF trajectory reports. Statistics for fields that only one schema stores come from a `SchemaDiagnostics` subclass (`diagnostics/schema_diagnostics.py`), whose generic parameter is the schema's `TrajectoryMetadata` subclass, or its `Trajectory` subclass when `compute_columns()` reads the steps (as `TunixDiagnostics` does with `TunixTrajectory`):

```python
class AcmeEvalDiagnostics(
    schema_diagnostics.SchemaDiagnostics[EvalTrajectoryMetadata]
):
  COLUMN_DTYPES = {"verdict": "string", "latency_ms": "Float64"}
```

- **Columns**: each column must use a nullable pandas dtype (`string`, `Int64`, `Float64` or `boolean`). In `RunStats.df` it becomes `{NAMESPACE}__{column}`, where the namespace comes from the schema's `EXTENSIONS_KEY` (`_acme_eval_extensions` becomes `acme_eval`). In the `summary` table it appears under its own name, so, unless `format_summary_columns()` renames it, it must not reuse a header already in the table (`trajectory`, `status`, `steps`, `agent`, `env`, `tokens`, `tool calls`, `reward`, or a column of another diagnostics): `summary` raises `ValueError` rather than overwrite that column.
- **Output**: `summary` adds the columns after `reward` (rows of other schemas show `-`) and `show` adds one `column: value` line to the header of each trajectory of the schema; lines from `format_summary_totals()` follow `Mean reward:`. Under `--json`, `summary` rows carry `acme_eval__verdict` and `acme_eval__latency_ms`, `show` reports the values under `stats.schema_stats["acme_eval"]`, and `summary` reports the run summary under `run.schema_stats["acme_eval"]`.
- **Hidden columns**: the default `format_summary_columns()` and `format_show_header()` leave out the columns listed in `HIDDEN_COLUMNS` (e.g. `HIDDEN_COLUMNS = frozenset({"latency_ms"})`, each a column of `COLUMN_DTYPES`). They are still extracted, so they reach `RunStats.df`, `compute_run_summary()` and `--json`: use them for values that the run summary needs but that are not worth printing per trajectory.
- **Hooks**: `compute_columns()` returns a record's value of each column; by default, it reads the record's attribute of the same name. `extract()` is final: it returns the result of `compute_columns()` after checking it against `COLUMN_DTYPES`. Override `compute_run_summary()` for the run summary (`schema_stats` in `--json`; the final `summarize()` checks that it is JSON serializable), and `format_show_header()`, `format_summary_columns()` or `format_summary_totals()` to change what `show` and `summary` print.
- **Activation**: nothing is registered. Pass instances to `TrajectoryStats.from_trajectory(..., diagnostics=...)` and `RunStats.from_trajectories(..., diagnostics=...)`, or to `run_explorer.launch_cli(diagnostics=...)` from a CLI entry point; the bundled `trajectory-explore` entry point passes the diagnostics of its own schema, `TunixDiagnostics` (`diagnostics/tunix_diagnostics.py`). A diagnostics class only applies to trajectories that are instances of its schema.

---

## How to Add a New Subcommand

Adding a new subcommand requires **4 steps**. `commands/summary.py` and `commands/show.py` are complete examples; shared formatting helpers (missing-value marker, optional values, counts) live in `diagnostics/formatting.py`, and `StepCounts`/`TokenCounts` render themselves via `str()`.

### Step 1: Implement `BaseCommand` in `commands/<name>.py`
Create a frozen dataclass inheriting from `base.BaseCommand`. Any fields declared on the dataclass automatically become subcommand-specific CLI flags via `simple_parsing`.

```python
@dataclasses.dataclass(frozen=True, kw_only=True)
class MyCommand(base.BaseCommand):
  """One-line description shown in --help."""

  some_flag: str = simple_parsing.field(help="Becomes --some_flag.")

  def execute(
      self,
      reader: store.TrajectoryReader,
      output_json: bool = False,
      *,
      diagnostics: Sequence[schema_diagnostics.SchemaDiagnostics] = (),
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
    cmd = show.ShowCommand(trajectory_id=["traj_1001"])
    exit_code, out = testing.execute_command(cmd, mem_store, output_json=True)
    self.assertEqual(exit_code, 0)
    self.assertIn("traj_1001", out)
```

### Step 4: Run Tests
```bash
pytest tests/dist_rl/trajectory/explorer/
```
