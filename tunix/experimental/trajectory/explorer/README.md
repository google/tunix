# Trajectory Explorer CLI (`trajectory-explore`)

`trajectory-explore` is a read-only diagnostic CLI for inspecting agentic RL rollouts and trajectories stored in a `TrajectoryStore`.

## Core Design Concepts

1. **Strict Read-Only Protocol Boundary (`TrajectoryReader`)**
   The CLI parses global store flags (`--backend`, `--root_dir`, `--run_id`) via `simple_parsing`, constructs the selected backend through `store.TrajectoryStore.from_config()`, and immediately narrows the instance to `store.TrajectoryReader`. Subcommands only ever receive a `TrajectoryReader`, guaranteeing that inspection tools cannot mutate or corrupt rollout data or couple to backend-specific storage internals.

2. **Dual Output Contract (Human & `--json`)**
   Every subcommand implements `execute(self, reader: store.TrajectoryReader, output_json: bool = False) -> int`.
   - When `output_json=False` (default), commands print formatted, human-readable terminal summaries using `termcolor`.
   - When `output_json=True` (`--json`), commands emit machine-readable JSON to `stdout` while any CLI usage errors are isolated to `stderr` (exit code `2`), enabling clean composition with `jq` and automated evaluation scripts.

3. **Decoupled Testing Architecture**
   CLI flag parsing and store initialization are tested once in `main_test.py`. Individual subcommand tests (`commands/*_test.py`) do **not** touch the filesystem or CLI parser; instead, they test command logic directly against prefilled in-memory stores using the test harness in `commands/testing.py`.

---

## Usage

### Human-Readable Output
```bash
python3 -m tunix.experimental.trajectory.explorer.main \
  --backend file \
  --root_dir /tmp/tx_smoke \
  --run_id run_95d297b1 \
  ping
```

### Machine-Readable JSON Output
```bash
python3 -m tunix.experimental.trajectory.explorer.main \
  --backend file \
  --root_dir /tmp/tx_smoke \
  --run_id run_95d297b1 \
  --json ping
# {"status": "ok", "trajectories_count": 4}
```

---

## How to Add a New Subcommand

Adding a new subcommand (e.g., `summary`, `show`, or `diff`) requires **4 steps**:

### Step 1: Implement `BaseCommand` in `commands/<name>.py`
Create a frozen dataclass inheriting from `base.BaseCommand`. Any fields declared on the dataclass automatically become subcommand-specific CLI flags via `simple_parsing`.

```python
import dataclasses
import json
import simple_parsing
from tunix.experimental.trajectory import store
from tunix.experimental.trajectory.explorer.commands import base


@dataclasses.dataclass(frozen=True, kw_only=True)
class ShowCommand(base.BaseCommand):
  """Displays the steps of a single trajectory."""

  trajectory_id: str = simple_parsing.field(
      help="Unique identifier of the trajectory to inspect."
  )

  def execute(
      self, reader: store.TrajectoryReader, output_json: bool = False
  ) -> int:
    trajectories = reader.get_trajectories([self.trajectory_id])
    if output_json:
      print(json.dumps([t.model_dump(mode="json") for t in trajectories]))
      return 0
    # Render human-readable output...
    return 0
```

### Step 2: Register in `commands/__init__.py`
Add your command class to `COMMAND_REGISTRY`:

```python
from tunix.experimental.trajectory.explorer.commands import show

COMMAND_REGISTRY: dict[str, type[base.BaseCommand]] = {
    "ping": ping.PingCommand,
    "show": show.ShowCommand,
}
```

### Step 3: Write Unit Tests in `commands/<name>_test.py`
Use `testing.create_store()` and `testing.execute_command()` to test both human-readable and JSON output modes against an in-memory store pre-populated with sample trajectories (`TRAJECTORY_1`, `TRAJECTORY_2`):

```python
from absl.testing import absltest
from tunix.experimental.trajectory.explorer.commands import show
from tunix.experimental.trajectory.explorer.commands import testing


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
pytest tests/experimental/trajectory/explorer/
```
