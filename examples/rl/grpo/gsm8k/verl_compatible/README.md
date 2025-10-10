# Training with verl compatible data and reward function

This example folder serves as a simple example to train with a verl compatible
setup, but on Tunix with TPU.

## Data preparation

To run this example, first run the
[gsm8k script](https://github.com/volcengine/verl/blob/main/examples/data_preprocess/gsm8k.py)
from verl to get the data prepared, and place the data folder in the following
way:

```
grpo
  |_data
    |_gsm8k
      |_train.parquet
      |_test.parquet
```

## Reward setup

There's a reward defined in `cli/reward_fn/gsm8k_verl.py`, and you can directly
start the training with this, which is copy-pasted from verl defined in 
https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/gsm8k.py,
or write your own reward function.

## Config setup

There's an example config defined in `examples/cli/rl/grpo/gsm8k/verl_compatible/run_llama3.2_1b.sh`. Feel free
to modify according to your own setting.

## Training

Run via:

```
./examples/cli/rl/grpo/gsm8k/verl_compatible/run_llama3.2_1b.sh
```