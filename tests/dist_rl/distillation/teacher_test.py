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

"""HTTP protocol, alignment, and scheduling tests for the OPD teacher scorer."""

import asyncio
import copy
import dataclasses
import http.server
import json
import threading
import time
import types
import unittest

import numpy as np
from tunix.dist_rl.distillation import teacher


@dataclasses.dataclass
class _AddedToken:
  content: str
  single_word: bool = False
  lstrip: bool = False
  rstrip: bool = False
  normalized: bool = False
  special: bool = True


class _Tokenizer:

  def __init__(self):
    self.vocab = {f"token_{index}": index for index in range(64)}
    self.added_vocab = {"token_63": 63}
    self.added_tokens_decoder = {63: _AddedToken("token_63")}
    self.bos_token_id = 1
    self.eos_token_id = 2
    self.pad_token_id = 0
    self.all_special_ids = [0, 1, 2, 63]
    self.special_tokens_map = {
        "bos_token": "token_1",
        "eos_token": "token_2",
        "pad_token": "token_0",
    }
    self.backend_tokenizer = types.SimpleNamespace(
        to_str=lambda: json.dumps({
            "model": {"type": "WordLevel", "vocab": self.vocab},
            "decoder": None,
        })
    )

  def get_vocab(self):
    return dict(self.vocab)

  def get_added_vocab(self):
    return dict(self.added_vocab)


def _payload(prompt=(1, 4), completion=(5, 6), mask=None):
  return types.SimpleNamespace(
      prompt_ids=np.asarray(prompt, dtype=np.int32),
      prompt_mask=np.ones(len(prompt), dtype=np.float32),
      completion_ids=np.asarray(completion, dtype=np.int32),
      completion_mask=np.ones(len(completion), dtype=np.float32)
      if mask is None
      else np.asarray(mask, dtype=np.float32),
      segment_ids=None,
  )


def _response(body):
  return {
      "meta_info": {
          "input_token_logprobs": [
              [None if position == 0 else -token_id / 10, token_id, "ignored"]
              for position, token_id in enumerate(body["input_ids"])
          ]
      }
  }


class _Server(http.server.ThreadingHTTPServer):
  daemon_threads = True

  def __init__(self):
    super().__init__(("127.0.0.1", 0), _Handler)
    self.lock = threading.Lock()
    self.requests = []
    self.active = 0
    self.peak = 0
    self.delay = 0.0
    self.stream_interval = 0.0
    self.status = 200
    self.respond = _response


class _Handler(http.server.BaseHTTPRequestHandler):

  def log_message(self, *args):
    del args

  def do_POST(self):  # pylint: disable=invalid-name
    server = self.server
    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
    with server.lock:
      server.requests.append((self.path, body))
      server.active += 1
      server.peak = max(server.peak, server.active)
    try:
      time.sleep(server.delay)
      content = json.dumps(server.respond(body)).encode("utf-8")
      self.send_response(server.status)
      self.send_header("Content-Type", "application/json")
      self.send_header("Content-Length", str(len(content)))
      self.end_headers()
      if server.stream_interval:
        for byte in content:
          self.wfile.write(bytes([byte]))
          self.wfile.flush()
          time.sleep(server.stream_interval)
      else:
        self.wfile.write(content)
    except (BrokenPipeError, ConnectionResetError):
      pass
    finally:
      with server.lock:
        server.active -= 1


class TeacherTokenizerTest(unittest.TestCase):

  def _scorer(self, student=None, teacher_tokenizer=None, **kwargs):
    return teacher.SGLangTeacherScorer(
        "http://localhost:30000",
        student_tokenizer=student or _Tokenizer(),
        teacher_tokenizer=teacher_tokenizer or _Tokenizer(),
        **kwargs,
    )

  def test_equal_tokenizers_have_stable_fingerprint(self):
    first = self._scorer()
    second = self._scorer()
    self.assertIsInstance(first, teacher.TeacherScorer)
    self.assertEqual(first.tokenizer_fingerprint, second.tokenizer_fingerprint)
    self.assertEqual(first.teacher_url, "http://localhost:30000/generate")

  def test_same_size_vocabulary_with_swapped_ids_is_rejected(self):
    tokenizer = _Tokenizer()
    tokenizer.vocab["token_4"], tokenizer.vocab["token_5"] = 5, 4
    with self.assertRaisesRegex(ValueError, "vocab differ"):
      self._scorer(teacher_tokenizer=tokenizer)

  def test_added_vocabulary_mismatch_is_rejected(self):
    tokenizer = _Tokenizer()
    tokenizer.added_vocab = {}
    with self.assertRaisesRegex(ValueError, "added_vocab differ"):
      self._scorer(teacher_tokenizer=tokenizer)

  def test_added_token_flags_must_match(self):
    tokenizer = _Tokenizer()
    tokenizer.added_tokens_decoder[63].lstrip = True
    with self.assertRaisesRegex(ValueError, "added_tokens differ"):
      self._scorer(teacher_tokenizer=tokenizer)

  def test_special_token_ids_must_match(self):
    tokenizer = _Tokenizer()
    tokenizer.pad_token_id = 2
    with self.assertRaisesRegex(ValueError, "special_ids differ"):
      self._scorer(teacher_tokenizer=tokenizer)

  def test_special_token_roles_must_match(self):
    tokenizer = _Tokenizer()
    tokenizer.special_tokens_map["bos_token"] = "token_2"
    with self.assertRaisesRegex(ValueError, "special_tokens differ"):
      self._scorer(teacher_tokenizer=tokenizer)

  def test_invalid_request_limits_are_rejected(self):
    for kwargs in (
        {"timeout_seconds": 0},
        {"timeout_seconds": float("nan")},
        {"max_concurrency": 0},
        {"max_concurrency": True},
    ):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        self._scorer(**kwargs)

  def test_unverifiable_slow_tokenizer_is_rejected(self):
    tokenizer = _Tokenizer()
    del tokenizer.backend_tokenizer
    with self.assertRaisesRegex(ValueError, "fast tokenizer backend"):
      self._scorer(teacher_tokenizer=tokenizer)

  def test_real_fast_tokenizer_decoder_must_match(self):
    from tokenizers import decoders
    from tokenizers import models
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "a": 1, "b": 2}
    student_backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    teacher_backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    teacher_backend.decoder = decoders.Replace("a", "b")
    student = PreTrainedTokenizerFast(
        tokenizer_object=student_backend, unk_token="[UNK]"
    )
    other = PreTrainedTokenizerFast(
        tokenizer_object=teacher_backend, unk_token="[UNK]"
    )
    self.assertEqual(student.get_vocab(), other.get_vocab())
    self.assertNotEqual(student.decode([1]), other.decode([1]))
    with self.assertRaisesRegex(ValueError, "backend differ"):
      self._scorer(student=student, teacher_tokenizer=other)

  def test_padding_and_truncation_runtime_state_do_not_change_semantics(self):
    from tokenizers import models
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    def tokenizer():
      return PreTrainedTokenizerFast(
          tokenizer_object=Tokenizer(
              models.WordLevel({"[UNK]": 0, "a": 1, "b": 2}, unk_token="[UNK]")
          ),
          unk_token="[UNK]",
      )

    student, other = tokenizer(), tokenizer()
    other.backend_tokenizer.enable_padding(
        length=8, pad_id=0, pad_token="[UNK]"
    )
    other.backend_tokenizer.enable_truncation(max_length=16)
    self._scorer(student=student, teacher_tokenizer=other)


class TeacherHttpTest(unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self):
    self.server = _Server()
    self.thread = threading.Thread(
        target=lambda: self.server.serve_forever(poll_interval=0.01),
        daemon=True,
    )
    self.thread.start()
    self.url = f"http://127.0.0.1:{self.server.server_port}/generate"

  async def asyncTearDown(self):
    await asyncio.to_thread(self.server.shutdown)
    self.server.server_close()
    self.thread.join(timeout=1.0)

  def _scorer(self, **kwargs):
    return teacher.SGLangTeacherScorer(
        self.url,
        student_tokenizer=_Tokenizer(),
        teacher_tokenizer=_Tokenizer(),
        **kwargs,
    )

  async def test_request_uses_original_tokens_and_returns_completion_only(self):
    result = await self._scorer().score([_payload(), _payload(completion=(7,))])

    np.testing.assert_allclose(result[0], [-0.5, -0.6])
    np.testing.assert_allclose(result[1], [-0.7])
    self.assertEqual(result[0].dtype, np.float32)
    path, body = next(
        request
        for request in self.server.requests
        if len(request[1]["input_ids"]) == 4
    )
    self.assertEqual(path, "/generate")
    self.assertEqual(body["input_ids"], [1, 4, 5, 6])
    self.assertEqual(body["logprob_start_len"], 0)
    self.assertTrue(body["return_logprob"])
    self.assertEqual(
        body["sampling_params"],
        {"max_new_tokens": 0, "temperature": 1.0, "skip_special_tokens": False},
    )

  async def test_entirely_masked_rows_do_not_contact_teacher(self):
    result = await self._scorer().score(
        [_payload(mask=(0, 0)), _payload(completion=(), mask=())]
    )
    np.testing.assert_array_equal(result[0], [0, 0])
    self.assertEqual(result[1].shape, (0,))
    self.assertEqual(self.server.requests, [])

  async def test_empty_group_is_supported(self):
    self.assertEqual(await self._scorer().score([]), [])

  async def test_masked_failed_tokens_do_not_fail_healthy_peers(self):
    result = await self._scorer().score([
        _payload(),
        _payload(prompt=(), completion=(-1, -1), mask=(0, 0)),
    ])
    np.testing.assert_allclose(result[0], [-0.5, -0.6])
    np.testing.assert_array_equal(result[1], [0, 0])
    self.assertEqual(len(self.server.requests), 1)

  async def test_bad_inputs_are_rejected_before_starting_requests(self):
    for name, value in (
        ("prompt_ids", np.array([[1, 4]], dtype=np.int32)),
        ("prompt_ids", np.array([1.0, 4.0])),
        ("prompt_mask", np.array([0, 1])),
        ("completion_mask", np.array([1])),
        ("completion_mask", np.array([1, np.nan])),
        ("segment_ids", np.array([1, 1, 1, 1])),
        ("completion_ids", np.array([99, 6], dtype=np.int32)),
    ):
      with self.subTest(name=name, value=value):
        invalid = _payload()
        setattr(invalid, name, value)
        with self.assertRaises(ValueError):
          await self._scorer().score([_payload(), invalid])
    self.assertEqual(self.server.requests, [])

  async def test_response_token_ids_are_validated_including_prompt(self):
    for position in (0, 1, 2, 3):
      with self.subTest(position=position):

        def respond(body):
          output = _response(body)
          output["meta_info"]["input_token_logprobs"][position][1] = 10
          return output

        self.server.respond = respond
        with self.assertRaisesRegex(ValueError, "token ID mismatch"):
          await self._scorer().score([_payload()])

  async def test_malformed_or_missing_logprobs_fail(self):
    valid = _response({"input_ids": [1, 4, 5, 6]})
    bad_responses = [
        {},
        {"meta_info": {}},
        {
            "meta_info": {
                "input_token_logprobs": valid["meta_info"][
                    "input_token_logprobs"
                ][:-1]
            }
        },
    ]
    for position in (2, 3):
      for bad_value in (None, float("nan"), float("inf"), "-0.1", True):
        body = copy.deepcopy(valid)
        body["meta_info"]["input_token_logprobs"][position][0] = bad_value
        bad_responses.append(body)
    for bad_row in (None, [], [-0.5], [-0.5, None], [-0.5, True]):
      body = copy.deepcopy(valid)
      body["meta_info"]["input_token_logprobs"][2] = bad_row
      bad_responses.append(body)
    for response in bad_responses:
      with self.subTest(response=response):
        self.server.respond = lambda body, response=response: response
        with self.assertRaises(ValueError):
          await self._scorer().score([_payload()])

  async def test_only_initial_placeholder_can_have_missing_id_or_nan(self):
    def respond(body):
      result = _response(body)
      result["meta_info"]["input_token_logprobs"][0] = [float("nan"), None]
      return result

    self.server.respond = respond
    result = await self._scorer().score([_payload()])
    np.testing.assert_allclose(result[0], [-0.5, -0.6])

  async def test_prompt_and_masked_scores_are_ignored_but_context_is_preserved(
      self,
  ):
    for ignored in (None, float("nan"), float("-inf")):
      with self.subTest(ignored=ignored):

        def respond(body):
          result = _response(body)
          rows = result["meta_info"]["input_token_logprobs"]
          rows[1][0] = ignored
          rows[2][0] = ignored
          return result

        self.server.respond = respond
        result = await self._scorer().score([_payload(mask=(0, 1))])
        np.testing.assert_allclose(result[0], [0.0, -0.6])
        self.assertEqual(self.server.requests[-1][1]["input_ids"], [1, 4, 5, 6])

  async def test_masked_scores_still_require_correct_token_ids(self):
    def respond(body):
      result = _response(body)
      result["meta_info"]["input_token_logprobs"][2] = [None, 9]
      return result

    self.server.respond = respond
    with self.assertRaisesRegex(ValueError, "token ID mismatch"):
      await self._scorer().score([_payload(mask=(0, 1))])

  async def test_first_completion_uses_its_own_causal_logprob(self):
    # SGLang's scheduler prefixes None and removes its final next-token score:
    # raw [-.12, -.34, -1.25, -2.75, -9] -> [None, -.12, -.34, -1.25, -2.75].
    self.server.respond = lambda body: {
        "meta_info": {
            "input_token_logprobs": [
                [None, 1, None],
                [-0.12, 9, None],
                [-0.34, 4, None],
                [-1.25, 63, None],
                [-2.75, 7, None],
            ]
        }
    }
    result = await self._scorer().score(
        [_payload(prompt=(1, 9, 4), completion=(63, 7))]
    )
    np.testing.assert_allclose(result[0], [-1.25, -2.75])

  def test_zero_completion_parser_returns_an_empty_vector(self):
    response = _response({"input_ids": [1, 4]})
    result = teacher.SGLangTeacherScorer._completion_logps(
        response, [1, 4], np.zeros(0, np.float32)
    )
    self.assertEqual(result.shape, (0,))

  async def test_http_failure_propagates(self):
    self.server.status = 503
    with self.assertRaisesRegex(RuntimeError, "HTTP status 503"):
      await self._scorer().score([_payload()])

  async def test_request_timeout_propagates(self):
    self.server.delay = 0.15
    with self.assertRaisesRegex(TimeoutError, "timed out"):
      await self._scorer(timeout_seconds=0.02).score([_payload()])

  async def test_slow_drip_response_has_a_total_deadline_and_closes_socket(
      self,
  ):
    self.server.stream_interval = 0.005
    start = time.monotonic()
    with self.assertRaisesRegex(TimeoutError, "timed out"):
      await self._scorer(timeout_seconds=0.04).score([_payload()])
    self.assertLess(time.monotonic() - start, 0.4)
    # Cancellation must terminate the connection; the server's complete body
    # would otherwise take over a second despite never hitting an idle timeout.
    for _ in range(100):
      with self.server.lock:
        if not self.server.active:
          break
      await asyncio.sleep(0.003)
    self.assertEqual(self.server.active, 0)

  async def test_waiting_for_a_request_slot_has_the_same_total_deadline(self):
    scorer = self._scorer(timeout_seconds=0.02, max_concurrency=1)
    await scorer._semaphore.acquire()
    try:
      with self.assertRaisesRegex(TimeoutError, "timed out"):
        await scorer.score([_payload()])
      self.assertEqual(self.server.requests, [])
    finally:
      scorer._semaphore.release()

  async def test_concurrency_is_bounded_without_blocking_event_loop(self):
    self.server.delay = 0.04
    progress = asyncio.Event()

    async def heartbeat():
      for _ in range(100):
        await asyncio.sleep(0.002)
        with self.server.lock:
          if self.server.active:
            progress.set()
            return

    tick = asyncio.create_task(heartbeat())
    result = await self._scorer(max_concurrency=2).score([_payload()] * 6)
    await tick
    self.assertTrue(progress.is_set())
    self.assertEqual(len(result), 6)
    self.assertEqual(self.server.peak, 2)

  async def test_concurrent_score_calls_share_the_request_limit(self):
    self.server.delay = 0.02
    scorer = self._scorer(max_concurrency=1)
    await asyncio.gather(
        scorer.score([_payload()] * 2), scorer.score([_payload()] * 2)
    )
    self.assertEqual(len(self.server.requests), 4)
    self.assertEqual(self.server.peak, 1)

  async def test_cancelled_score_does_not_start_queued_requests(self):
    self.server.delay = 0.08
    scorer = self._scorer(max_concurrency=1)
    task = asyncio.create_task(scorer.score([_payload()] * 5))
    for _ in range(100):
      if self.server.requests:
        break
      await asyncio.sleep(0.002)
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task
    await asyncio.sleep(0.1)
    self.assertEqual(len(self.server.requests), 1)
    self.assertEqual(self.server.peak, 1)


if __name__ == "__main__":
  unittest.main()
