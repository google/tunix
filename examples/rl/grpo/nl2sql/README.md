# NL2SQL GRPO Example

This example trains a model to translate natural language questions into SQL
over a small SQLite database using GRPO. Rewards are based on execution success
and exact result matching.

## Setup

Build the SQLite database:

```
python3 examples/rl/grpo/nl2sql/build_db.py
```

## Train

Run GRPO with the provided config:

```
python3 -m tunix.cli.grpo_main \
  examples/rl/grpo/nl2sql/configs/base_config.yaml
```

## Notes

- The model should output a single SQL SELECT statement.
- The reward function is in `tunix/cli/reward_fn/nl2sql.py`.

