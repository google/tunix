# DeepSWE Evaluation on Tunix

## Overview

This directory contains scripts for running SWE-bench evaluation using
tunix's JAX-based inference, mirroring rllm's `run_deepswe.py`.

## How SWE-bench Evaluation Works

Each instance in the SWE-Bench-Verified dataset is a real GitHub issue
paired with a snapshot of the repository at the time the bug existed.
The agent's job is to fix the bug.

### Walkthrough: A Single Instance

Consider a dataset entry like the following:

```json
{
  "instance_id": "django__django-12345",
  "repo": "django/django",
  "problem_statement": "QuerySet.count() raises TypeError when database is empty...",
  "docker_image": "swebench/django__django-12345:latest",
  "FAIL_TO_PASS": ["tests/queries/test_count.py::TestCount::test_empty_db_count"],
  "PASS_TO_PASS": [
    "tests/queries/test_count.py::TestCount::test_normal_count",
    "tests/queries/test_count.py::TestCount::test_filtered_count"
  ]
}
```

**Step 1 -- Container startup.**
A Docker container (or K8s Pod) is created from `docker_image`. Inside:

- `/testbed/django/` contains the buggy repo checkout.
- `/run_tests.sh` is a pre-injected script that runs the relevant tests:

  ```bash
  #!/bin/bash
  set -uxo pipefail
  source /opt/miniconda3/bin/activate
  conda activate testbed
  cd /testbed/django/
  : '>>>>> Start Test Output'
  python -m pytest tests/queries/test_count.py -x -v
  : '>>>>> End Test Output'
  ```

**Step 2 -- Agent interaction.**
The agent receives the `problem_statement` and interacts with the
container through XML function calls. A typical session:

```
Agent: "Let me look at the count() implementation."
  -> file_editor view /testbed/django/db/models/query.py [280, 310]

Container: (shows source code of count())

Agent: "Found it -- line 295 does result[0] but result is None on an empty DB."
  -> file_editor str_replace /testbed/django/db/models/query.py
       old_str: "return result[0]"
       new_str: "return result[0] if result else 0"

Container: "File updated successfully."

Agent: "Let me verify the fix."
  -> execute_bash "cd /testbed/django && python -m pytest tests/queries/test_count.py -v"

Container: "3 passed"

Agent: "Fix verified."
  -> finish submit
```

The agent has four tools available: `file_editor`, `execute_bash`,
`search`, and `finish`.

**Step 3 -- Reward computation.**
After the agent finishes (or hits the step/time limit), the framework
runs `/run_tests.sh` inside the same container. The SWE-bench harness
parses the test output and checks two conditions:

| Check | Meaning | Requirement |
|-------|---------|-------------|
| FAIL\_TO\_PASS | Tests that *were* failing due to the bug | Must now PASS |
| PASS\_TO\_PASS | Tests that *were* already passing | Must still PASS |

Both conditions met -> `reward = 1` (resolved).
Either condition violated -> `reward = 0`.

Example of a successful fix:

```
FAIL_TO_PASS:
  test_empty_db_count:  FAILED -> PASSED  (ok)

PASS_TO_PASS:
  test_normal_count:    PASSED -> PASSED  (ok)
  test_filtered_count:  PASSED -> PASSED  (ok)

resolution_status = FULL -> reward = 1
```

Example of a regression (reward = 0 even though the target bug is fixed):

```
FAIL_TO_PASS:
  test_empty_db_count:  FAILED -> PASSED  (ok)

PASS_TO_PASS:
  test_normal_count:    PASSED -> FAILED  (regression!)
  test_filtered_count:  PASSED -> PASSED  (ok)

resolution_status != FULL -> reward = 0
```

**Step 4 -- Cleanup.**
The container is destroyed after reward computation.

### End-to-End Flow

```
SWE-Bench-Verified dataset (~500 instances)
  |
  |  For each instance (in parallel, up to MAX_CONCURRENT):
  |
  v
Start container  ->  Agent interacts (up to MAX_STEPS)  ->  Run /run_tests.sh
                        |                                       |
                        |  file_editor / execute_bash /         |  SWE-bench harness
                        |  search / finish                      |  parses test output
                        |                                       |
                        v                                       v
                   Agent submits fix               reward = 1 or 0
                                                        |
                                                   Close container
                                                        |
                                                        v
                                              Aggregate Pass@1
```

## Files

| File | Description |
|------|-------------|
| `eval_deepswe.py` | Main evaluation script |
| `swe_agent.py` | Agent that parses XML function calls and maintains conversation |
| `swe_env.py` | Environment wrapper around R2E-Gym's RepoEnv |
| `setup.sh` | Creates conda environment and installs dependencies |

## Usage

```bash
# 1. Setup environment
bash setup.sh /path/to/workdir

# 2. Activate
conda activate deepswe_eval

# 3. Run evaluation (small test)
TASKS_LIMIT=2 MAX_CONCURRENT=1 python eval_deepswe.py

# 4. Run full evaluation
TASKS_LIMIT=0 MAX_CONCURRENT=16 python eval_deepswe.py
```

### Configuration (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_NAME` | `R2E-Gym/SWE-Bench-Verified` | HuggingFace dataset |
| `DATASET_SPLIT` | `test` | Dataset split |
| `MODEL_VERSION` | `Qwen/Qwen3-4B-Instruct-2507` | Model to evaluate |
| `MAX_STEPS` | `30` | Max agent steps per instance |
| `MAX_CONCURRENT` | `8` | Parallel instances |
| `TIMEOUT` | `600` | Per-instance timeout (seconds) |
| `TASKS_LIMIT` | `10` | Number of instances (0 = all) |
| `OUTPUT_DIR` | `/scratch/eval_results` | Where to save results |
