# Copyright 2026 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A deadline must not strand other users of the shared native sampler."""

import asyncio
import dataclasses
from types import SimpleNamespace

import pytest
from examples.swarm_math import train


@dataclasses.dataclass
class Request:
  request_id: str


@pytest.mark.parametrize("failure", [False, True])
def test_cancelled_queued_and_inflight_requests_do_not_strand_peers(failure):
  async def run():
    started, release = asyncio.Event(), asyncio.Event()
    observed = []

    async def sample(requests, **kwargs):
      del kwargs
      observed.extend(r.request_id for r in requests)
      started.set()
      await release.wait()
      if failure:
        raise RuntimeError("serving failed")
      return [r.request_id for r in requests]

    sampler = object.__new__(train.NativeMathSampler)
    sampler.batch_size = 3
    sampler.config = SimpleNamespace(max_prompt_length=16)
    sampler.sampler = SimpleNamespace(sample=sample)
    loop = asyncio.get_running_loop()
    queued, inflight, peer = [loop.create_future() for _ in range(3)]
    queued.cancel()
    sampler._queue = list(
        zip(
            [Request("queued"), Request("inflight"), Request("peer")],
            [queued, inflight, peer],
            strict=True,
        )
    )
    task = asyncio.create_task(sampler._flush_queued())
    await started.wait()
    inflight.cancel()
    release.set()
    await asyncio.wait_for(task, timeout=1)
    if failure:
      with pytest.raises(RuntimeError, match="serving failed"):
        await peer
    else:
      assert await peer == "peer"
    assert "queued" not in observed
    assert queued.cancelled() and inflight.cancelled()
    assert not sampler._queue

  asyncio.run(run())
