# Copyright 2026 The Tunix Authors.
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

"""Sampler backed by the continuous batching engine."""

import itertools
from typing import Any, Optional, Sequence, Tuple

from absl import logging
from flax import nnx
from flax.nnx import graph
from flax.nnx import statelib
import jaxtyping
import numpy as np
from tunix.experimental.generate import driver as driver_lib
from tunix.experimental.generate import engine as engine_lib
from tunix.experimental.generate import request as request_lib
from tunix.generate import base_sampler
from tunix.generate import tokenizer_adapter as tok_adapter
from tunix.generate import utils as generate_utils

# Truncation is a property of the config, not of any one prompt, so the warning
# is worth seeing but not worth repeating once per request.
_MAX_TRUNCATION_LOGS = 5


def _tokenizer_eos_token_ids(tokenizer: Any) -> tuple[int, ...] | None:
  """Returns the tokenizer's EOS token ids, or None if it names none."""
  if not hasattr(tokenizer, 'eos_id'):
    return None
  eos = tokenizer.eos_id()
  if eos is None:
    return None
  return tuple(eos) if isinstance(eos, (list, tuple, set)) else (eos,)


class Sampler(base_sampler.BaseSampler):
  """Sampler for transformer model."""

  def __init__(
      self,
      transformer: nnx.Module,
      tokenizer: Any,
      engine_config: engine_lib.EngineConfig,
      enable_server_mode: bool = True,
      poll_interval_s: float = 0.004,
      submission_threshold: int = 0,
      submission_timeout_s: float = 0.0,
  ):
    """Initializes the sampler.

    Args:
      transformer: an instance of the transformer.
      tokenizer: a tokenizer for the given model.
      engine_config: configuration for engine.
      enable_server_mode: wheter or not to enable server mode. In server mode
        the engine runs on a background thread, so prompts submitted by
        separate callers share a continuous batch. Otherwise each call drives
        the engine itself and only batches the prompts it was given.
      poll_interval_s: How long the background loop waits for new work before
        looking again. Ignored if server mode is disabled.
      submission_threshold: Only hand queued requests to the engine once this
        many have accumulated. 0 submits them as soon as they arrive. Ignored
        if server mode is disabled.
      submission_timeout_s: Submit a partial batch anyway once this many
        seconds have elapsed since the first request of the current window
        arrived. 0 disables the timeout. Ignored if server mode is disabled.
    """
    if not isinstance(tokenizer, tok_adapter.TokenizerAdapter):
      tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
    self._tokenizer = tokenizer
    self._engine_config = engine_config
    # EOS lives on the request, and the sampler is what builds requests, so the
    # tokenizer's EOS is resolved once here and stamped onto every request a
    # caller does not name one for. Without this a caller that omits EOS would
    # generate until `max_tokens_to_generate`, which looks like a hang.
    self._default_eos_token_ids = _tokenizer_eos_token_ids(tokenizer)
    self._engine = engine_lib.LLMEngine(
        transformer=transformer,
        engine_config=engine_config,
    )
    self._request_ids = itertools.count()

    self._driver: Optional[driver_lib.VanillaInProcessDriver] = None
    if enable_server_mode:
      self._driver = driver_lib.VanillaInProcessDriver(
          transformer=transformer,
          engine=self._engine,
          poll_interval_s=poll_interval_s,
          submission_threshold=submission_threshold,
          submission_timeout_s=submission_timeout_s,
      )
      self._driver.start()

  @property
  def engine(self) -> engine_lib.LLMEngine:
    return self._engine

  @property
  def driver(self) -> Optional[driver_lib.VanillaInProcessDriver]:
    """The driver running the engine, or None outside of server mode."""
    return self._driver

  @property
  def transformer(self) -> nnx.Module:
    return self._engine.transformer

  @property
  def transformer_state(self) -> statelib.State:
    return self._engine.transformer_state

  @property
  def tokenizer(self) -> Any:
    return self._tokenizer

  def model_def_and_state(self) -> tuple[graph.GraphDef[Any], list[Any]]:
    """Returns the transformer graphdef and state."""
    return self._engine.model_def_and_state()

  def pad_id(self) -> int:
    """Returns the tokenizer's padding token ID."""
    if hasattr(self._tokenizer, 'pad_id'):
      return self._tokenizer.pad_id()
    if (
        hasattr(self._tokenizer, 'pad_token_id')
        and self._tokenizer.pad_token_id is not None
    ):
      return int(self._tokenizer.pad_token_id)
    return 0

  def eos_id(self) -> int:
    """Returns the tokenizer's EOS token ID."""
    if hasattr(self._tokenizer, 'eos_id'):
      return self._tokenizer.eos_id()
    if (
        hasattr(self._tokenizer, 'eos_token_id')
        and self._tokenizer.eos_token_id is not None
    ):
      return int(self._tokenizer.eos_token_id)
    return 1

  def tokenize(self, input_string: str) -> np.ndarray:
    """Tokenizes a prompt, truncated to the engine's `max_prompt_length`.

    Args:
      input_string: The prompt to tokenize.

    Returns:
      The prompt's token ids, at most `max_prompt_length` of them.
    """
    input_ids = self._tokenizer.encode(input_string)

    bos_id = (
        self._tokenizer.bos_id() if hasattr(self._tokenizer, 'bos_id') else None
    )
    if bos_id is not None:
      bos_tok = [bos_id] if bos_id else []
      if hasattr(self._tokenizer, 'dedup_bos_ids'):
        input_ids = np.array(
            self._tokenizer.dedup_bos_ids(bos_tok + input_ids), dtype=np.int32
        )
      else:
        input_ids = np.array(bos_tok + input_ids, dtype=np.int32)
    else:
      input_ids = np.array(input_ids, dtype=np.int32)

    return self._truncate_prompt(input_ids, bos_id)

  def _truncate_prompt(
      self, token_ids: np.ndarray, bos_id: int | None
  ) -> np.ndarray:
    """Trims a prompt down to `max_prompt_length` tokens.

    The engine sizes its per-request page table, and `_to_sampler_output` its
    prompt padding, off `max_prompt_length`, so a longer prompt cannot be run
    as-is. The tail is kept, since that is the part the model is asked to
    continue, along with the leading BOS the model expects.

    Args:
      token_ids: The prompt's token ids.
      bos_id: The tokenizer's BOS token id, or None if it has none.

    Returns:
      The prompt, at most `max_prompt_length` tokens long.
    """
    max_prompt_length = self._engine_config.max_prompt_length
    if len(token_ids) <= max_prompt_length:
      return token_ids

    logging.log_first_n(
        logging.WARNING,
        'Truncating a prompt of %d tokens to max_prompt_length=%d. Raise'
        ' max_prompt_length to keep the whole prompt.',
        _MAX_TRUNCATION_LOGS,
        len(token_ids),
        max_prompt_length,
    )

    if bos_id is not None and token_ids[0] == bos_id and max_prompt_length > 1:
      return np.concatenate(
          [token_ids[:1], token_ids[-(max_prompt_length - 1) :]]
      )
    return token_ids[-max_prompt_length:]

  def update_params(
      self,
      updated_weights: jaxtyping.PyTree,
      filter_types: Optional[Tuple[Any, ...]] = None,
  ) -> None:
    """Syncs new weights into the model this sampler generates with.

    Args:
      updated_weights: The new weights.
      filter_types: The variable types to update. All of them when None.
    """
    if self._driver is not None:
      # Goes through the driver so that the update lands between steps rather
      # than underneath the loop thread.
      self._driver.update_params(updated_weights, filter_types)
    else:
      self._engine.update_params(updated_weights, filter_types)

  def shutdown(self) -> None:
    """Stops the background loop, if there is one."""
    if self._driver is not None:
      self._driver.shutdown()

  def __call__(
      self,
      input_strings: str | Sequence[str],
      max_tokens_to_generate: int | Sequence[int | None] | None = None,
      eos_token_ids: Sequence[int] | set[int] | None = None,
      **kwargs,
  ) -> base_sampler.SamplerOutput:
    """Samples a continuation for each of the input strings.

    Args:
      input_strings: The prompt, or prompts, to sample from.
      max_tokens_to_generate: Optional per-request token limit (or scalar for
        all).
      eos_token_ids: Optional per-request EOS token IDs.
      **kwargs: Additional keyword arguments. Accepts 'max_generation_steps' and
        'eos_tokens' as aliases for max_tokens_to_generate and eos_token_ids.

    Returns:
      The generated samples, in the order the prompts were given.
    """
    if max_tokens_to_generate is None and 'max_generation_steps' in kwargs:
      max_tokens_to_generate = kwargs.pop('max_generation_steps')
    if eos_token_ids is None and 'eos_tokens' in kwargs:
      eos_token_ids = kwargs.pop('eos_tokens')
    if eos_token_ids is None:
      eos_token_ids = self._default_eos_token_ids

    if isinstance(input_strings, str):
      input_strings = [input_strings]
    # Tokenization, and the truncation that goes with it, happens here so that
    # the engine and driver only ever see token ids that already fit.
    prompt_token_ids = [
        [int(t) for t in self.tokenize(prompt)] for prompt in input_strings
    ]

    if self._driver is not None:
      finished = [
          future.result()
          for future in self._driver.submit_requests(
              prompt_token_ids,
              max_tokens_to_generate=max_tokens_to_generate,
              eos_token_ids=eos_token_ids,
          )
      ]
    else:
      finished = self._generate(
          prompt_token_ids,
          max_tokens_to_generate=max_tokens_to_generate,
          eos_token_ids=eos_token_ids,
      )

    self._engine.log_perf()  # PERF-DEBUG

    return self._to_sampler_output(finished)

  def _generate(
      self,
      prompt_token_ids: list[list[int]],
      max_tokens_to_generate: int | Sequence[int | None] | None = None,
      eos_token_ids: Sequence[int] | set[int] | None = None,
  ) -> list[request_lib.Request]:
    """Drives the engine directly until every prompt has finished."""
    submitted = []
    for i, token_ids in enumerate(prompt_token_ids):
      if (
          isinstance(max_tokens_to_generate, int)
          or max_tokens_to_generate is None
      ):
        max_tokens = max_tokens_to_generate
      else:
        max_tokens = max_tokens_to_generate[i]
      req = request_lib.Request(
          f'sampler_{id(self)}_{next(self._request_ids)}',
          list(token_ids),
          max_tokens_to_generate=max_tokens,
          eos_token_ids=eos_token_ids,
      )
      self._engine.add_request(req)
      submitted.append(req.request_id)

    finished_by_id: dict[str, request_lib.Request] = {}
    while self._engine.has_unfinished_requests():
      for req in self._engine.step():
        finished_by_id[req.request_id] = req

    return [finished_by_id[request_id] for request_id in submitted]

  def _to_sampler_output(
      self, finished: list[request_lib.Request]
  ) -> base_sampler.SamplerOutput:
    """Reshapes finished requests into the output samplers hand back."""
    config = self._engine_config
    # Padding to a power of 2 keeps downstream compilations shape-stable.
    # `tokenize` truncates to `max_prompt_length`, so this width holds every
    # prompt and does not move from batch to batch.
    padded_prompt_length = generate_utils.next_power_of_2(
        config.max_prompt_length
    )

    texts = []
    tokens = []
    padded_prompt_tokens = []
    logits = [] if config.return_logits else None
    logprobs = [] if config.return_logprobs else None

    for req in finished:
      all_tokens = np.array(req.token_ids, dtype=np.int32)
      prompt_tokens = all_tokens[: req.prompt_length]
      out_tokens = all_tokens if config.echo else all_tokens[req.prompt_length :]

      tokens.append(out_tokens)
      padded_prompt_tokens.append(
          generate_utils.pad_to_length(
              prompt_tokens,
              target_length=padded_prompt_length,
              pad_value=self.pad_id(),
              left=True,
          )
      )
      texts.append(self._decode(out_tokens))

      if logits is not None:
        logits.append(np.array(req.logits))
      if logprobs is not None:
        logprobs.append([float(lp) for lp in req.logprobs])

    return base_sampler.SamplerOutput(
        text=texts,
        logits=logits,
        tokens=tokens,
        padded_prompt_tokens=np.array(padded_prompt_tokens),
        logprobs=logprobs,
    )

  def _decode(self, tokens: np.ndarray) -> str:
    token_list = (
        tokens.tolist() if isinstance(tokens, np.ndarray) else list(tokens)
    )
    if hasattr(self._tokenizer, 'decode'):
      try:
        return self._tokenizer.decode(token_list, skip_special_tokens=True)
      except TypeError:
        return self._tokenizer.decode(token_list)
    return ''.join(str(t) for t in tokens)

