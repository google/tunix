# Copyright 2026 Google LLC
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
"""Unit tests for Qwen3.5-9B config and full-paper R2Orch JAX/TPU modules."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
import tempfile
from typing import Any
from unittest import mock

from absl.testing import absltest
from flax import nnx
import jax.numpy as jnp
import numpy as np
from tunix.generate import base_sampler
from tunix.models import naming
from tunix.models import registry
from tunix.models.qwen3p5 import mapping_vllm_jax
from tunix.models.qwen3p5 import model as qwen3p5_model

from examples.swarm_r2orch import paper_eval
from examples.swarm_r2orch import paper_kernels
from examples.swarm_r2orch import paper_lora
from examples.swarm_r2orch import paper_retrieval
from examples.swarm_r2orch import paper_serving
from examples.swarm_r2orch import paper_trainer


def _make_tiny_qwen3p5() -> qwen3p5_model.Qwen3p5:
  cfg = qwen3p5_model.ModelConfig(
      num_layers=4,
      vocab_size=64,
      embed_dim=32,
      hidden_dim=64,
      num_heads=4,
      head_dim=16,
      num_kv_heads=2,
      rope_theta=10_000_000,
      norm_eps=1e-6,
      use_tied_embedding=False,
      use_norm_offset=True,
      attn_output_gate=True,
      partial_rotary_factor=0.25,
      full_attention_interval=4,
      linear_num_key_heads=4,
      linear_num_value_heads=4,
      linear_key_head_dim=8,
      linear_value_head_dim=8,
      linear_conv_kernel_dim=4,
      remat_config=qwen3p5_model.RematConfig.DECODER,
      dtype=jnp.float32,
      param_dtype=jnp.float32,
      use_flash_attention=False,
  )
  return qwen3p5_model.Qwen3p5(cfg, rngs=nnx.Rngs(7))


class _TinyDummyTokenizer:
  eos_token_id = 2

  def convert_tokens_to_ids(self, tok: str) -> int:
    return {"<|im_end|>": 2, "<|endoftext|>": 1}.get(tok, 3)

  def apply_chat_template(
      self,
      messages: list[dict[str, Any]],
      *,
      chat_template: str = "",
      tokenize: bool = False,
      add_generation_prompt: bool = True,
      **kwargs: Any,
  ) -> str:
    del chat_template, tokenize, kwargs
    body = "|".join(f"{m['role']}:{m['content']}" for m in messages)
    return body + ("|assistant:" if add_generation_prompt else "")

  def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
    del add_special_tokens
    table = {"off": 10, "on": 11, "A": 20, "B": 21}
    if text in table:
      return [table[text]]
    return [4 + (ord(ch) % 50) for ch in text[:24]] or [4]

  def decode(
      self,
      ids: list[int],
      skip_special_tokens: bool = False,
      clean_up_tokenization_spaces: bool = False,
  ) -> str:
    del skip_special_tokens, clean_up_tokenization_spaces
    rev = {10: "off", 11: "on", 20: "A", 21: "B"}
    if len(ids) == 1 and ids[0] in rev:
      return rev[ids[0]]
    return "".join(chr(97 + (i % 26)) for i in ids)


class _DummyVllmSampler:

  def __init__(self) -> None:
    self.updated = 0

  def update_params(self, weights: Any) -> None:
    preprocessed = mapping_vllm_jax.preprocess_src_state(weights, tp_size=1)
    assert preprocessed is not None
    self.updated += 1

  def reset_prefix_cache(self) -> None:
    return None

  def __call__(
      self,
      *,
      prompt_token_ids: list[list[int]],
      max_generation_steps: int,
      temperature: float = 0.0,
      **kwargs: Any,
  ) -> base_sampler.SamplerOutput:
    del max_generation_steps, temperature, kwargs
    assert prompt_token_ids
    tokens = [np.asarray([12, 13, 2], dtype=np.int32)]
    logprobs = [[-0.5, -0.4, -0.1]]
    return base_sampler.SamplerOutput(
        text=["mn"],
        logits=None,
        tokens=tokens,
        padded_prompt_tokens=np.asarray(prompt_token_ids, dtype=np.int32),
        logprobs=logprobs,
    )


@dataclass
class _StubPolicyConfig:
  checkpoint_root: Any
  learning_rate: float = 1e-3
  warmup_steps: int = 1
  total_optimizer_steps: int = 10
  learning_rate_schedule: str = "cosine"
  max_grad_norm: float = 1.0
  proposer_max_grad_norm: float = 0.0
  proposer_optimizer_steps: int = 1
  ppo_mini_batches: int = 2
  proposer_kl_beta: float = 1.0
  proposer_gap_clip: float = 4.0
  weight_decay: float = 0.01
  lora_rank: int = 4
  lora_alpha: float = 4.0
  kl_coefficient: float = 0.005
  kl_reference_stride: int = 2
  kl_term_cap: float = 10.0
  max_sequence_length: int = 1024
  raw_policy_backward_mode: str = "timeline"
  fuse_old_logprobs: bool = True
  clip_low: float = 0.2
  clip_high: float = 0.28
  tis_cap: float = 2.0
  epochs: int = 1
  publish_latest_checkpoint: bool = True


@dataclass
class _StubCall:
  call_id: str
  token_ids: tuple[int, ...]
  action_mask: tuple[int, ...]
  behavior_log_probs: tuple[float, ...]
  relation_token_span: tuple[int, int] | None = None
  metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StubSample:
  rollout_id: str
  task_id: str
  token_ids: tuple[int, ...]
  action_mask: tuple[int, ...]
  reward: float
  advantage: float
  policy_calls: tuple[_StubCall, ...] = ()
  metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _StubBatch:
  role: str
  objective: str
  samples: tuple[_StubSample, ...]
  metadata: dict[str, Any] = field(default_factory=dict)


class R2OrchPaperStackTest(absltest.TestCase):

  def test_qwen3p5_9b_model_config_and_registry(self):
    cfg = qwen3p5_model.ModelConfig.qwen3p5_9b()
    self.assertEqual(cfg.num_layers, 32)
    self.assertEqual(cfg.vocab_size, 248320)
    self.assertEqual(cfg.embed_dim, 4096)
    self.assertEqual(cfg.hidden_dim, 12288)
    self.assertEqual(cfg.num_heads, 16)
    self.assertEqual(cfg.num_kv_heads, 4)
    self.assertFalse(cfg.use_tied_embedding)
    self.assertEqual(cfg.linear_num_value_heads, 32)
    info = naming.ModelNaming(model_id="Qwen/Qwen3.5-9B")
    self.assertEqual(info.model_config_id, "qwen3p5_9b")
    catalog_ids = {m.model_config_id for m in registry.MODEL_CATALOG}
    self.assertIn("qwen3p5_9b", catalog_ids)

  def test_bucket_padding_causal_invariance_and_lora_merge(self):
    model = _make_tiny_qwen3p5()
    hub = paper_trainer.SharedTunixLearnerHub(None, preloaded_model=model)
    seq = [5, 9, 14, 21, 7, 18, 30]
    unpadded_tokens = jnp.asarray([seq], dtype=jnp.int32)
    unpadded_pos = jnp.arange(len(seq), dtype=jnp.int32)[None, :]
    unpadded_seg = jnp.ones((1, len(seq)), dtype=jnp.int32)
    direct = paper_kernels.forward_selected_logits(
        hub.graphdef,
        hub.base_state,
        unpadded_tokens,
        unpadded_pos,
        unpadded_seg,
        jnp.asarray([2, 6], dtype=jnp.int32),
    )
    bucketed = hub.eval_selected_logits(seq, [2, 6], reference=True)
    np.testing.assert_allclose(np.asarray(direct), bucketed, atol=1e-4, rtol=1e-4)

    lora_p = paper_lora.init_lora_params(hub.base_state, rank=4, seed=3)
    merged = paper_lora.merge_lora_state(hub.base_state, lora_p, scale=1.0)
    vllm_ready = mapping_vllm_jax.preprocess_src_state(merged, tp_size=1)
    self.assertIsNotNone(vllm_ready)

  def test_gateway_constrained_and_chat_endpoints(self):
    model = _make_tiny_qwen3p5()
    hub = paper_trainer.SharedTunixLearnerHub(None, preloaded_model=model)
    tok = _TinyDummyTokenizer()
    dummy_wrk = _DummyVllmSampler()
    dummy_dir = _DummyVllmSampler()
    with tempfile.TemporaryDirectory() as tmp:
      cfg = _StubPolicyConfig(checkpoint_root=tmp)
      hub.ensure_role("solver", cfg, seed=1)
      hub.ensure_role("proposer", cfg, seed=2)
      gw = paper_serving.R2OrchTpuGatewayState(
          hf_tokenizer=tok,
          director_jinja="",
          hub=hub,
          worker_sampler=dummy_wrk,
          director_sampler=dummy_dir,
          max_model_len=512,
      )
      bin_out = gw.handle_chat_completion({
          "model": "r2orch-proposer",
          "messages": [{"role": "user", "content": "pick?"}],
          "temperature": 1.0,
          "seed": 9,
          "allowed_token_ids": [20, 21],
          "logprob_token_ids": [20, 21],
      })
      self.assertIn(bin_out["choices"][0]["message"]["content"], {"A", "B"})
      self.assertEqual(len(bin_out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]), 2)

      chat_out = gw.handle_chat_completion({
          "model": "r2orch-director",
          "messages": [{"role": "user", "content": "act"}],
          "temperature": 0.7,
          "max_tokens": 16,
          "logprobs": True,
      })
      self.assertEqual(chat_out["choices"][0]["finish_reason"], "stop")
      self.assertEqual(chat_out["choices"][0]["token_ids"], [12, 13, 2])
      gw.load_adapter("r2orch-director-v0002", tmp)
      self.assertEqual(dummy_dir.updated, 1)
      self.assertGreaterEqual(len(gw.recent_traces), 2)

  def test_director_timeline_and_relation_update(self):
    model = _make_tiny_qwen3p5()
    hub = paper_trainer.SharedTunixLearnerHub(None, preloaded_model=model)
    # Call 0: ordinary tokens at indices 3, 4 (predicting token_ids[4], token_ids[5])
    c0_ids = (4, 5, 6, 7, 12, 13)
    c0_mask = (0, 0, 0, 0, 1, 1)
    c0 = _StubCall("c0", c0_ids, c0_mask, (0.0, 0.0, 0.0, -3.5, -3.5))
    # Call 1: exact causal extension with binary `off/on` relation token at index 8
    c1_ids = (4, 5, 6, 7, 12, 13, 8, 9, 11)
    c1_mask = (0, 0, 0, 0, 0, 0, 0, 0, 1)
    c1 = _StubCall(
        "c1",
        c1_ids,
        c1_mask,
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.7),
        relation_token_span=(8, 9),
        metadata={"relation_policy": {"token_ids": {"off": 10, "on": 11}}},
    )
    s1 = _StubSample(
        "r1",
        "t1",
        c1_ids,
        c1_mask,
        1.0,
        0.8,
        policy_calls=(c0, c1),
        metadata={"r2orch": {"scaffold_condition": "on", "kl_coefficient": 0.005}},
    )
    s2 = _StubSample(
        "r2",
        "t1",
        c1_ids,
        c1_mask,
        0.0,
        -0.8,
        policy_calls=(c0, c1),
        metadata={"r2orch": {"scaffold_condition": "off", "kl_coefficient": 0.005}},
    )
    batch = _StubBatch("solver", "r2orch_director", (s1, s2))

    fake_r2_tl = mock.MagicMock()
    fake_r2_tl.timeline_positions.side_effect = lambda calls, limit: [[3, 4], [7]]
    fake_r2_tr = mock.MagicMock()
    fake_r2_tr.PolicyUpdateResult = lambda *a, **kw: {"args": a, "kwargs": kw}
    with (
        tempfile.TemporaryDirectory() as tmp,
        mock.patch.dict(
            "sys.modules",
            {
                "r2orch": mock.MagicMock(),
                "r2orch.policy_timeline": fake_r2_tl,
                "r2orch.training": fake_r2_tr,
            },
        ),
    ):
      cfg = _StubPolicyConfig(checkpoint_root=tmp)
      view = hub.factory("solver", cfg, seed=5)
      res = view.update(batch, step=1)
      self.assertEqual(res["kwargs"]["optimizer_steps"], 2)
      self.assertGreater(res["kwargs"]["grad_norm"], 0.0)

  def test_sharded_wiki18_search_kernel(self):
    import jax  # pylint: disable=g-import-not-at-top

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]), ("ret",))
    search_fn = paper_retrieval._build_sharded_search_fn(mesh)
    mat = np.zeros((16, paper_retrieval.EMBED_DIM), dtype=np.float32)
    mat[7, 0] = 1.0
    mat[3, 1] = 1.0
    ids = np.arange(16, dtype=np.int32)
    ids[12:] = -1
    q = np.zeros((2, paper_retrieval.EMBED_DIM), dtype=np.float32)
    q[0, 0] = 1.0
    q[1, 1] = 1.0
    _, top_ids = search_fn(
        jnp.asarray(q),
        jnp.asarray(mat, dtype=jnp.bfloat16),
        jnp.asarray(ids),
    )
    top_np = np.asarray(top_ids)
    self.assertEqual(int(top_np[0, 0]), 7)
    self.assertEqual(int(top_np[1, 0]), 3)
    self.assertTrue(np.all(top_np[:, :12] >= 0))

  def test_pooled_zero_skill_evaluator_and_deadline(self):
    @dataclass
    class _FakeCanvas:
      submission_journal_dir: str = ""

    @dataclass
    class _FakeSolverModel:
      served_model: str = "r2orch-base"

    @dataclass
    class _FakePaths:
      evalplus_python: str = ""

    @dataclass
    class _FakeR2:
      paths: _FakePaths = field(default_factory=_FakePaths)

    @dataclass
    class _FakeCfg:
      seed: int = 0
      skillbank_enabled: bool = True
      trace_path: Any = None
      canvas: _FakeCanvas = field(default_factory=_FakeCanvas)
      solver_model: _FakeSolverModel = field(default_factory=_FakeSolverModel)
      r2orch: _FakeR2 = field(default_factory=_FakeR2)

    deadlines_set: list[Any] = []
    closed_count = [0]

    class _FakeApp:

      def __init__(self) -> None:
        self.solver = mock.MagicMock()

      def set_rollout_deadline(self, dl: Any) -> None:
        deadlines_set.append(dl)

      def solve(self, prompt: str, **kwargs: Any) -> Any:
        del prompt, kwargs
        res = mock.MagicMock()
        res.to_dict.return_value = {
            "answer": "\\boxed{42}",
            "finished": True,
            "final_graph": {"nodes": {"a1": {}, "a2": {}}},
        }
        return res

      def close(self) -> None:
        closed_count[0] += 1

    fake_obs = mock.MagicMock()
    fake_obs.TaskSpec = lambda **kw: mock.MagicMock(**kw, private_verifier_payload=None)
    fake_dl = mock.MagicMock()
    fake_dl.RolloutDeadline = lambda **kw: kw
    fake_app_mod = mock.MagicMock()
    fake_app_mod.create_adaptive_application = lambda *a, **kw: _FakeApp()
    with (
        tempfile.TemporaryDirectory() as tmp,
        mock.patch.dict(
            "sys.modules",
            {
                "r2orch": mock.MagicMock(),
                "r2orch.observability": fake_obs,
                "r2orch.deadline": fake_dl,
                "r2orch.application": fake_app_mod,
            },
        ),
    ):
      p_math = f"{tmp}/math_hard.jsonl"
      with open(p_math, "w", encoding="utf-8") as f:
        f.write('{"id": "mh1", "prompt": "6*7?", "kind": "math_hard", "answer": "42"}\n')
      rep = paper_eval.evaluate_zero_skill_suites(
          _FakeCfg(),
          _TinyDummyTokenizer(),
          {"math_hard": p_math},
          output_dir=f"{tmp}/out",
          served_director="r2orch-director-v0002",
          concurrency=4,
      )
      self.assertEqual(rep["mean_acc"], 100.0)
      self.assertEqual(rep["suites"]["math_hard"]["multi_agent_pct"], 100.0)
      self.assertEqual(len(deadlines_set), 1)
      self.assertEqual(deadlines_set[0]["total_timeout_s"], 1800.0)
      self.assertEqual(closed_count[0], 1)


if __name__ == "__main__":
  absltest.main()
