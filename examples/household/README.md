# Household Task Exploration (AlfWorld / TextWorld style)

This recipe provides a multi-turn, interactive text environment and agentic reinforcement learning training pipeline using **Tunix Agentic RL**.

Inspired by benchmarks like **ALFWorld** and **TextWorld**, the environment tests an LLM agent's capacity for:
1. **Multi-room Spatial Navigation:** Exploring rooms, traversing directional exits, and mapping house layouts.
2. **Container & State Interaction:** Opening, closing, inspecting, and unlocking containers (e.g. drawers, cabinets, safes, chests).
3. **Multi-step Object Manipulation:** Finding keys, unlocking targets, picking up objects, and putting items into target containers.
4. **Chain-of-Thought (CoT) Reasoning:** Thinking step-by-step about current state and deciding on structured text actions.

---

## Architecture Overview

- **`env.py` (`HouseholdEnv`):** Inherits from `tunix.rl.agentic.environments.base_environment.BaseTaskEnv`. Manages room graph, container state transitions, inventory tracking, and milestone reward signals (`GOAL_REWARD=1.0`, milestone increments `0.1` for finding keys, opening target containers, unlocking, taking/placing items, and small step/stagnation penalties).
- **`agent.py` (`HouseholdAgent`):** Inherits from `tunix.rl.agentic.agents.base_agent.ConversationAgentBase`. Formats multi-turn chat history with a system prompt and parses actions enclosed in triple backticks (````action\n<command>\n``` ` or ````\n<command>\n``` `). Handles anti-stagnation warnings when the agent repeats invalid actions or produces identical states.
- **`data.py`:** Procedurally generates deterministic household exploration tasks (`take`, `put`, `open`) and converts them into Grain `MapDataset` instances. Supports passing raw task dictionaries or JSON-serialized records to the environment.
- **`train_household.py`:** End-to-end GRPO training loop using `tunix.rl.agentic.agentic_grpo_learner.AgenticGRPOLearner`. Supports **Gemma 4** (`google/gemma-4-E2B-it`) as the default first-class model using `Gemma4ChatTemplateParser(enable_thinking=False)`, as well as Qwen 3 (`Qwen/Qwen3-8B`).
- **`household_test.py`:** Hermetic unit test suite verifying environment transitions, natural language variations, unlocking, inventory, agent parsing, and task generator determinism.

---

## Action Verbs

The agent interacts with the environment using natural language commands:

| Action Verb | Syntax Example | Description |
|---|---|---|
| **Move** | `go north` / `go to kitchen` / `go to the bedroom` | Move to adjacent room. Multi-word room names and articles are supported. |
| **Look** | `look` | Inspect current room, exits, and visible objects. |
| **Inventory** | `inventory` | Show list of carried items. |
| **Open** | `open kitchen drawer` / `open the kitchen drawer` | Open a container in the current room. |
| **Close** | `close refrigerator` | Close an open container. |
| **Unlock** | `unlock wooden chest with silver key` | Unlock a container using a key in inventory. |
| **Take** | `take silver key from drawer` / `take the silver key` | Take item from container or floor. |
| **Put** | `put apple in refrigerator` | Place carried item into container. |
| **Examine** | `examine wooden chest` / `examine the safe` | Inspect container lock status and contents. |

---

## Running Unit Tests

Run the unit tests:

```bash
pytest examples/household/household_test.py
```

---

## Verification with Dummy Weights (Local CPU / GPU / TPU)

To verify the training loop, JAX tracer compilation, sharding, and rollout engine without requiring physical TPU hardware or large model downloads, run the script with `--dummy_weights`:

```bash
python3 -m examples.household.train_household \
  --dummy_weights \
  --batch_size=2 \
  --mini_batch_size=2 \
  --num_batches=1 \
  --num_generations=2 \
  --max_prompt_length=1024 \
  --max_response_length=32 \
  --max_steps=2
```

---

## Training with GRPO (TPU v5p / v6e)

### Single TPU Host Execution

Launch full GRPO training with Gemma 4 (`google/gemma-4-E2B-it`):

```bash
./examples/household/run_household.sh "google/gemma-4-E2B-it" 16 100
```

Or configure directly via the CLI:

```bash
python3 -m examples.household.train_household \
  --model_id="google/gemma-4-E2B-it" \
  --batch_size=16 \
  --mini_batch_size=16 \
  --num_generations=4 \
  --learning_rate=1e-6 \
  --num_batches=100
```

### Preconfigured YAML Configurations

Preconfigured YAML experiment profiles are available under `configs/`:
- `configs/gemma4_e2b.yaml`: Default Gemma 4 configuration (`google/gemma-4-E2B-it`).
- `configs/qwen3_8b.yaml`: Qwen 3 baseline (`Qwen/Qwen3-8B`).
