# Training with verl compatible data and reward function

This example folder serves as a simple example to train with a verl compatible
setup, but on tunix with TPU.

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

There's a dummy reward defined in `reward_fn/gsm8k.py`, and you can directly
start the training with this.

However, to properly train the model, you can either copy-paste the reward
function from verl defined in
https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/gsm8k.py,
or write your own reward function.

## Config setup

There's an example config defined in `config/llama3p2_1b_gsm8k.yaml`. Feel free
to modify according to your own setting.

## Training

First navigate to the folder `examples/grpo`, then run via:

```
python main_train.py --config=config/llama3p2_1b_gsm8k.yaml
```
