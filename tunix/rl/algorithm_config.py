# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import dataclasses
from absl import logging

@dataclasses.dataclass(slots=True, kw_only=True)
class AlgorithmConfig:
  """Configuration for RL algorithms.

  Parameters:
    algo_variant: The core algorithm variant to use.
    advantage_estimator: The advantage estimator to use.
    policy_loss_fn: The policy loss function to use.
  """

  algo_variant: str = "grpo"
  advantage_estimator: str = "grpo"
  policy_loss_fn: str = "grpo"
  reward_manager: str = "sequence-level"
  # Optional symmetric clamp applied to per-token KL inside
  # `common.compute_kl_divergence`. `None` (default) disables the clamp and
  # preserves prior behavior bit-for-bit. Set to a positive float (e.g.
  # `10000.0`) to bound rare outliers — useful when the trained policy
  # briefly drifts far from the reference and the `low_var_kl` estimator's
  # `exp(diff)` term saturates bf16 / overflows fp32 and poisons the loss
  # for the rest of the step.
  kl_clamp_value: float | None = None

  # --- Sampler-vs-trainer divergence -----------------------------------------
  # The rollout engine and the trainer run the same weights through different
  # kernels, so their log-probabilities differ slightly. Everything below is
  # off by default and reproduces prior behaviour exactly when unset. All of it
  # requires the rollout engine to return log-probabilities, and none of it is
  # supported under sequence packing.
  #
  # Drop rollout-truncated sequences from the update. They are a prefix of a
  # trajectory rather than a trajectory, so the reward attached to them is not
  # the reward for the behaviour that produced them. Removes them from the loss
  # AND its denominator, so survivors keep their full gradient magnitude and
  # only the effective batch size shrinks.
  overlong_loss_masking: bool = False
  # Drop sequences whose multiplicative probability error --
  # `mean_t exp(|log trainer_t - log sampler_t|)` -- exceeds this. 1.0 is
  # perfect agreement; 2.0 means the typical token's probability differs by a
  # factor of two, which indicates a fault rather than ordinary drift. Read
  # `sample_mask/mult_prob_error_{mean,max}` before choosing a value.
  seq_logprob_error_threshold: float | None = None
  # Apply a truncated importance-sampling correction rather than only reporting
  # the divergence. "seq-mask-tis" weights each token by its raw
  # sampler-to-trainer ratio and zeroes the weights of any sequence whose
  # geometric-mean ratio leaves the band below.
  #
  # NOTE dropped sequences stay in the loss denominator, so the loss -- and the
  # gradient -- scale down with the drop fraction. That is the opposite of
  # `overlong_loss_masking`. Watch `tis/is_oob_ratio`: a high value there is a
  # silent learning-rate cut, not merely a smaller batch.
  truncated_importance_sampling_type: str | None = None
  truncated_importance_sampling_ratio_min: float | None = None
  truncated_importance_sampling_ratio: float | None = None
  # Keep-bands to report a would-be drop rate for, without applying them, as
  # `sampler_is/would_drop_<lo>_<hi>`. Lets a band be chosen from data before
  # it is switched on. No band is assumed; empty disables the reporting.
  sampler_is_report_bands: tuple[tuple[float, float], ...] = ()
  # Inclusive upper bounds, in completion tokens, of the length buckets used by
  # the `sampler_is/lenscale/*` diagnostic; one open-ended bucket is appended.
  # Bucketing the sequences already in a batch by their own length measures how
  # the per-sequence sampler-vs-trainer offset scales with sequence length,
  # which distinguishes iid token noise (shrinks as 1/sqrt(T)) from a
  # systematic within-sequence bias (does not). Choose edges that split the
  # completion-length distribution into comparably populated bins.
  sampler_is_length_buckets: tuple[int, ...] | None = None

  def validate_sampler_is_options(self):
    """Validates the sampler-vs-trainer options above.

    A separate method rather than inline in `__post_init__` because subclasses
    override `__post_init__` without chaining to it, so inline validation would
    silently not run for them.

    Raises:
      ValueError: If an option is set to an unsupported or incomplete value.
    """
    lo = self.truncated_importance_sampling_ratio_min
    hi = self.truncated_importance_sampling_ratio
    if (lo is None) != (hi is None):
      raise ValueError(
          "truncated_importance_sampling_ratio_min and"
          " truncated_importance_sampling_ratio must be set together (a"
          f" keep-band needs both ends). Got min={lo}, max={hi}."
      )
    if lo is not None and hi is not None and lo > hi:
      raise ValueError(
          "truncated_importance_sampling_ratio_min must not exceed"
          f" truncated_importance_sampling_ratio. Got min={lo}, max={hi}."
      )
    if self.truncated_importance_sampling_type is not None:
      if self.truncated_importance_sampling_type != "seq-mask-tis":
        raise ValueError(
            "truncated_importance_sampling_type only supports 'seq-mask-tis'."
            f" Received: {self.truncated_importance_sampling_type!r}"
        )
      if lo is None:
        raise ValueError(
            "truncated_importance_sampling_type requires a keep-band. Set"
            " truncated_importance_sampling_ratio_min and"
            " truncated_importance_sampling_ratio."
        )
    for band in self.sampler_is_report_bands or ():
      if len(band) != 2 or band[0] > band[1]:
        raise ValueError(
            "each entry of sampler_is_report_bands must be an ordered"
            f" (min, max) pair. Received: {band!r}"
        )
    if self.sampler_is_length_buckets is not None:
      edges = tuple(self.sampler_is_length_buckets)
      if not edges:
        raise ValueError(
            "sampler_is_length_buckets must be non-empty when set; use None"
            " to disable the length-scaling diagnostic."
        )
      if any(e <= 0 for e in edges) or any(
          b <= a for a, b in zip(edges, edges[1:])
      ):
        raise ValueError(
            "sampler_is_length_buckets must be strictly increasing positive"
            f" token counts. Received: {edges}"
        )
      self.sampler_is_length_buckets = edges
    if self.sampler_is_report_bands:
      self.sampler_is_report_bands = tuple(
          tuple(b) for b in self.sampler_is_report_bands
      )

  def __post_init__(self):
    valid_algo_variants = [
        "grpo",
        "gspo-token",
        "ppo",
        "dapo",
    ]
    # "rloo" and "drgrpo" are registered estimators that this list previously
    # rejected, so they were unreachable on any config that runs this
    # validation. Widening the list only permits what the registry already
    # implements.
    valid_advantage_estimators = ["grpo", "grpo-loo", "rloo", "drgrpo", "gae"]
    valid_policy_loss_fns = ["grpo", "ppo"]
    if self.algo_variant not in valid_algo_variants:
      raise ValueError(
          f"algo_variant must be one of {valid_algo_variants}. "
          f"Received: {self.algo_variant!r}"
      )
    if self.advantage_estimator not in valid_advantage_estimators:
      raise ValueError(
          f"advantage_estimator must be one of {valid_advantage_estimators}."
          f" Received: {self.advantage_estimator}"
      )
    if self.policy_loss_fn not in valid_policy_loss_fns:
      raise ValueError(
          f"policy_loss_fn must be one of {valid_policy_loss_fns}."
          f" Received: {self.policy_loss_fn}"
      )
    self.validate_sampler_is_options()

    # Automatically prints configuration upon initialization.
    self.print_config()

  def print_config(self):
    """Prints all configuration fields, working dynamically for child classes."""
    logging.info(f"Initializing {self.__class__.__name__}:")
    for field in dataclasses.fields(self):
      value = getattr(self, field.name)
      logging.info(f"  {field.name}: {value}")
