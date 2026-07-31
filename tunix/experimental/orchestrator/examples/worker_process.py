# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runs one worker in its own process, owning its own model state.

Every distributed run in this tree so far has been several servers in one
process sharing a single cluster object, which exercises the call path but not
the thing that makes distribution hard: separate processes cannot see each
other's memory, so weights have to be transported, configuration has to agree
without a shared object to read, and a loss has to be described rather than
handed over.

This is the entry point for the other side of that boundary. Launched as

    python -m tunix.experimental.orchestrator.examples.worker_process \
        --role=trainer --port=50051 --transport_dir=/tmp/run

it constructs its own model state, serves the worker on a port, and prints a
line when it is ready so a launcher can wait for it rather than sleeping.
"""

import asyncio
from collections.abc import Sequence
from typing import Any

from absl import app
from absl import flags
from absl import logging
import numpy as np

from tunix.experimental.orchestrator import hosted_rollout_worker
from tunix.experimental.orchestrator import startup_validation
from tunix.experimental.orchestrator import worker_discovery
from tunix.experimental.orchestrator import trainer_handle as trainer_handle_lib
from tunix.experimental.orchestrator import weight_transport
from tunix.experimental.testing import toy_trainer
from tunix.experimental.worker import remote_execution

_ROLE = flags.DEFINE_string(
    "role", "trainer", "Which worker to serve: trainer or rollout."
)
_PORT = flags.DEFINE_integer("port", 0, "Port to serve on.")
_TRANSPORT_DIR = flags.DEFINE_string(
    "transport_dir", "", "Shared directory weights are staged through."
)
_VOCAB_SIZE = flags.DEFINE_integer("vocab_size", 16, "Toy vocabulary size.")
_GRAD_ACCUMULATION_STEPS = flags.DEFINE_integer(
    "grad_accumulation_steps", 1, "Micro-batches per optimizer update."
)
_TOKENIZER_HASH = flags.DEFINE_string(
    "tokenizer_hash", "toy-vocab", "Identifies the vocabulary in use."
)

READY_MARKER = "WORKER_READY"


class ToyRolloutEngine:
  """A sampler whose output depends on the weights it currently holds.

  Deliberately crude: the point is not the quality of the generation but that
  the tokens change when new weights arrive, which is what makes a weight
  transport observable from the outside.
  """

  def __init__(self, vocab_size: int):
    self._vocab_size = vocab_size
    self.weights: dict[str, np.ndarray] = {
        "w": np.zeros((vocab_size,), dtype=np.float32)
    }
    self.generations = 0

  def generate(self, prompts, *args, **kwargs):
    del args, kwargs
    self.generations += 1
    # The highest-scoring token under the current weights, so a weight change
    # shows up in the output.
    best = int(np.argmax(self.weights["w"]))
    tokens = [
        np.array([best, (best + 1) % self._vocab_size], dtype=np.int32)
        for _ in prompts
    ]
    return _Output(prompts, tokens)

  def update_weights(self, metadata) -> None:
    """Fetches the staged weights this sync round points at."""
    coordinates = getattr(metadata, "source_metadata", None)
    if coordinates is None:
      return
    transport = weight_transport.FileWeightTransport(
        _TRANSPORT_DIR.value or "."
    )
    self.weights = dict(transport.fetch(coordinates))
    logging.info("Rollout installed weight version %d", coordinates.version)


class _Output:
  """The shape batched generation returns."""

  def __init__(self, prompts, tokens):
    self.text = [f"completion {index}" for index in range(len(prompts))]
    self.tokens = tokens
    self.logprobs = [
        np.zeros(len(token), dtype=np.float32) for token in tokens
    ]
    self.left_padded_prompt_tokens = np.array(
        [[0, 1] for _ in prompts], dtype=np.int32
    )
    self.logits = None


def build_worker(role: str, transport_dir: str, vocab_size: int):
  """Constructs the worker this process owns.

  Args:
    role: "trainer" or "rollout".
    transport_dir: Shared directory for staged weights.
    vocab_size: Toy vocabulary size.

  Returns:
    The worker instance to serve.

  Raises:
    ValueError: On an unknown role.
  """
  if role == "trainer":
    return trainer_handle_lib.AbstractTrainerHandle(
        toy_trainer.ToyAbstractTrainer(
            {"vocab_size": vocab_size, "learning_rate": 0.5}
        ),
        grad_accumulation_steps=_GRAD_ACCUMULATION_STEPS.value,
        worker_id="trainer",
        transport=weight_transport.FileWeightTransport(transport_dir),
    )
  if role == "rollout":
    return hosted_rollout_worker.HostedRolloutWorker(
        ToyRolloutEngine(vocab_size), worker_id="rollout"
    )
  raise ValueError(f"Unknown worker role: {role!r}")


def serve(worker, port: int, context: Any = None) -> None:
  """Serves `worker` until the process is killed.

  The blocking entry point cannot be used here: it does not return until the
  server stops, so there would be no moment at which to announce readiness.

  Args:
    worker: The worker to serve.
    port: Port to listen on.
    context: Optional runtime context. When given, the worker announces itself
      through discovery once it is listening, which is how an orchestrator
      learns where it is instead of being configured with the address.
  """
  server = remote_execution.GrpcRemoteExecutionServer(worker)
  loop = asyncio.new_event_loop()
  asyncio.set_event_loop(loop)
  loop.run_until_complete(server.start_serving_async(port))

  # Announced only after the port accepts connections. A listener builds a
  # handle the moment it hears, and a handle to a socket that is not up yet
  # fails on first use.
  if context is not None:
    worker_discovery.announce(
        context.ipc.discovery,
        worker_discovery.WorkerAnnouncement(
            role=_ROLE.value,
            worker_id=worker.info().worker_id,
            port=port,
            resources=_declared_resources(),
        ),
    )
  print(f"{READY_MARKER} {port}", flush=True)
  loop.run_forever()


def _declared_resources() -> dict[str, Any]:
  """What this worker states about its configuration, for the fleet check."""
  return startup_validation.describe_resources(
      tokenizer_hash=_TOKENIZER_HASH.value,
      pad_id=0,
      eos_id=1,
      vocab_size=_VOCAB_SIZE.value,
  )


def main(argv: Sequence[str], context: Any = None) -> None:
  """Entry point, taking the runtime's context when launched through it."""
  del argv
  serve(
      build_worker(_ROLE.value, _TRANSPORT_DIR.value, _VOCAB_SIZE.value),
      _PORT.value,
      context,
  )


if __name__ == "__main__":
  app.run(main)
