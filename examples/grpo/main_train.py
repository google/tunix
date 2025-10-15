import importlib
import os

from absl import app
from absl import flags
from flax import nnx
from tunix.rl import rl_cluster as rl_cluster_lib
from utils import data as data_utils
from utils import model as model_utils
from utils import train as train_utils
import yaml

flags.DEFINE_string("config", None, "The path to the config file.")
FLAGS = flags.FLAGS


def make_reward_fn(reward_fn_path):
  module_name = os.path.splitext(os.path.basename(reward_fn_path))[0]
  spec = importlib.util.spec_from_file_location(module_name, reward_fn_path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)

  def reward_fn(prompts, completions, reward_model, **kwargs):
    del prompts, kwargs
    ground_truths = reward_model["ground_truth"]
    return [
        module.compute_score(c, gt) for c, gt in zip(completions, ground_truths)
    ]

  return reward_fn


def main(argv):
  del argv  # Unused.
  with open(FLAGS.config, "r") as f:
    config = yaml.safe_load(f)
  model, tokenizer, mesh = model_utils.get_model(config["model"])
  actor = model
  if config["model"].get("lora", None):
    actor = model_utils.get_lora_model(
        model, **config["model"]["lora"], model_mesh=mesh
    )
  nnx.display(actor)
  dataset = data_utils.get_dataset_from_parquet(
      config["data"]["train_data"], tokenizer
  ).batch(config["train"]["batch_size"])
  optimizer = train_utils.get_optimizer(config["train"])
  cluster_config = train_utils.get_rl_cluster_config(
      config["train"], mesh, optimizer
  )
  rl_cluster = rl_cluster_lib.RLCluster(
      actor=actor,
      reference=model,
      tokenizer=tokenizer,
      cluster_config=cluster_config,
  )
  reward_fns = [
      make_reward_fn(reward_fn_path)
      for reward_fn_path in config["train"]["reward_fns"]
  ]
  trainer = train_utils.get_trainer(config["train"], rl_cluster, reward_fns)
  with mesh:
    trainer.train(dataset)


if __name__ == "__main__":
  app.run(main)
