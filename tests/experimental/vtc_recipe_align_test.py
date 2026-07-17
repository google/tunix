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

"""Recipe-alignment pins: experimental/vtc_{data,reward}.py == agentic demo.

The standard `grpo_main` CLI path is aligned to the converging agentic demo
recipe (examples/math_gsm8k/qwen3_grpo_demo.py) via two plug-in modules and
yaml overrides (see tasks/recipe_align_two_paths/). These tests pin the
alignment:

  * the VTC prompt template is byte-identical to the demo's;
  * the ported reward scores EQUAL the demo's `_vtc_completion_outcome` on a
    fixture battery (demo functions are AST-extracted and exec'd, so the demo
    module's heavy jax/vllm imports are never executed);
  * the yaml carries every override and the same ROPE hot-patch as the
    verified gsm8k_refactor_stream.yaml.

Only stdlib + numpy are required; vtc_data.py itself is not imported (it
needs grain/tfds), its pure helpers are AST-extracted the same way.
"""

import ast
import importlib.util
import inspect
import os
import re

from absl.testing import absltest
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_DEMO = 'examples/math_gsm8k/qwen3_grpo_demo.py'
_VTC_DATA = 'experimental/vtc_data.py'
_VTC_REWARD = 'experimental/vtc_reward.py'
_ALIGN_YAML = 'experimental/qwen3_1p7b_grpo.yaml'
_STREAM_YAML = 'experimental/gsm8k_refactor_stream.yaml'


def _read(rel_path):
  with open(os.path.join(_ROOT, rel_path), encoding='utf-8') as f:
    return f.read()


def _module_tree(rel_path):
  return ast.parse(_read(rel_path))


def _get_assign(tree, name):
  """The literal value of a top-level `name = <literal>` assignment."""
  for node in tree.body:
    if isinstance(node, ast.Assign):
      for target in node.targets:
        if isinstance(target, ast.Name) and target.id == name:
          return ast.literal_eval(node.value)
  raise KeyError(name)


def _exec_functions(tree, names, extra_globals=None):
  """Exec only the named top-level functions; heavy module imports never run."""
  namespace = {'re': re, 'np': np, 'Any': object}
  namespace.update(extra_globals or {})
  for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in names:
      exec(ast.unparse(node), namespace)  # pylint: disable=exec-used
  missing = [n for n in names if n not in namespace]
  if missing:
    raise KeyError(f'functions not found: {missing}')
  return namespace


def _load_vtc_reward():
  """Imports vtc_reward the same way the CLI reward loader does (file path)."""
  path = os.path.join(_ROOT, _VTC_REWARD)
  spec = importlib.util.spec_from_file_location('vtc_reward', path)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


# Fixture battery: covers all 4 score tiers plus the failure modes that
# motivated the alignment (Qwen3 <think> output, missing tags, nested boxed,
# comma numbers, bare-\boxed fallback, multiple answer blocks, empty).
_COMPLETIONS_AND_GOLDS = [
    ('steps</reasoning>\n<answer>\\boxed{42}</answer>', '42'),        # 1.0
    ('steps</reasoning>\n<answer>\\boxed{41}</answer>', '42'),        # 0.1
    ('the answer is \\boxed{42}', '42'),                              # 0.5
    ('i dont know', '42'),                                            # 0.0
    ('<think>hmm</think> the answer is 42', '42'),                    # thinking-mode output
    ('x</reasoning><answer>\\boxed{\\frac{84}{2}}</answer>', '42'),   # nested braces
    ('y</reasoning><answer>\\boxed{1,234}</answer>', '1,234'),        # commas both sides
    ('z</reasoning><answer>\\boxed 7</answer>', '7'),                 # bare fallback regex
    ('a</reasoning><answer>\\boxed{1}</answer><answer>\\boxed{2}</answer>', '2'),
    ('', '42'),                                                       # empty
    ('r</reasoning><answer></answer> \\boxed{42}', '42'),             # empty answer block
]


class VtcRewardTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.vtc_reward = _load_vtc_reward()

  def test_four_tier_known_answers(self):
    comps = [c for c, _ in _COMPLETIONS_AND_GOLDS[:4]]
    golds = [g for _, g in _COMPLETIONS_AND_GOLDS[:4]]
    scores = self.vtc_reward.vtc_reward(
        prompts=None, completions=comps, answer=golds
    )
    self.assertEqual(scores, [1.0, 0.1, 0.5, 0.0])

  def test_scores_equal_demo_on_fixture_battery(self):
    """The ported scorer == the demo's `_vtc_completion_outcome` (score)."""
    demo_ns = _exec_functions(
        _module_tree(_DEMO),
        [
            'extract_boxed_answer',
            'is_vtc_format_correct',
            'normalize_answer',
            '_normalize_example_value',
            '_vtc_completion_outcome',
        ],
    )
    for completion, gold in _COMPLETIONS_AND_GOLDS:
      demo_score = demo_ns['_vtc_completion_outcome'](completion, gold)[0]
      ported = self.vtc_reward.vtc_reward(
          prompts=None, completions=[completion], answer=[gold]
      )[0]
      self.assertEqual(
          ported,
          demo_score,
          msg=f'score mismatch for completion={completion!r} gold={gold!r}',
      )

  def test_gold_unwrapping_bytes_and_ndarray(self):
    ok = 'steps</reasoning>\n<answer>\\boxed{42}</answer>'
    for gold in (b'42', np.array([b'42']), np.array(['42'])):
      self.assertEqual(
          self.vtc_reward.vtc_reward(
              prompts=None, completions=[ok], answer=[gold]
          ),
          [1.0],
          msg=f'gold={gold!r}',
      )

  def test_exactly_one_public_function(self):
    """The CLI loader registers every public function: there must be one."""
    public = [
        name
        for name, fn in inspect.getmembers(self.vtc_reward, inspect.isfunction)
        if not name.startswith('_') and fn.__module__ == 'vtc_reward'
    ]
    self.assertEqual(public, ['vtc_reward'])


class VtcDataTest(absltest.TestCase):

  def test_template_byte_identical_to_demo(self):
    demo = _get_assign(_module_tree(_DEMO), 'VTC_PROMPT_TEMPLATE')
    ported = _get_assign(_module_tree(_VTC_DATA), 'VTC_PROMPT_TEMPLATE')
    self.assertEqual(ported, demo)

  def test_template_preopens_reasoning(self):
    template = _get_assign(_module_tree(_VTC_DATA), 'VTC_PROMPT_TEMPLATE')
    self.assertTrue(template.rstrip('\n').endswith('<reasoning>'))

  def test_build_prompt_matches_demo(self):
    demo_tree = _module_tree(_DEMO)
    demo_ns = _exec_functions(
        demo_tree,
        ['build_prompt'],
        extra_globals={
            'VTC_PROMPT_TEMPLATE': _get_assign(demo_tree, 'VTC_PROMPT_TEMPLATE')
        },
    )
    data_tree = _module_tree(_VTC_DATA)
    data_ns = _exec_functions(
        data_tree,
        ['_build_prompt'],
        extra_globals={
            'VTC_PROMPT_TEMPLATE': _get_assign(data_tree, 'VTC_PROMPT_TEMPLATE')
        },
    )
    question = 'Natalia sold clips to 48 of her friends in April. How many?'
    self.assertEqual(
        data_ns['_build_prompt'](question), demo_ns['build_prompt'](question)
    )

  def test_extract_hash_answer_matches_demo(self):
    demo_ns = _exec_functions(_module_tree(_DEMO), ['extract_hash_answer'])
    data_ns = _exec_functions(_module_tree(_VTC_DATA), ['_extract_hash_answer'])
    for text in (
        'reasoning #### 42',
        'no marker here',
        'a #### 1,234',  # demo keeps commas; reward normalizes at compare time
        'x #### 7 #### 8',
        '####   spaced   ',
    ):
      self.assertEqual(
          data_ns['_extract_hash_answer'](text),
          demo_ns['extract_hash_answer'](text),
          msg=f'text={text!r}',
      )


class YamlAlignmentTest(absltest.TestCase):
  """Pins the recipe-alignment overrides in qwen3_1p7b_grpo.yaml."""

  def setUp(self):
    super().setUp()
    self.yaml_text = _read(_ALIGN_YAML)

  def test_overrides_present(self):
    for needle in (
        'data_module="experimental/vtc_data.py"',
        'apply_chat_template_to_dataset=false',
        'reward_functions="[\'experimental/vtc_reward.py\']"',
        'rollout_config.total_generation_steps=1024',
        'rollout_config.max_prompt_length=1024',
        'rl_training_config.actor_optimizer_config.warmup_steps=50',
        'rl_training_config.actor_optimizer_config.decay_steps=500',
        'train_fraction=1.0',
        'rollout_engine="vllm"',
        'vllm_config.max_num_seqs=512',
        'vllm_config.max_num_batched_tokens=147456',
        'vllm_config.kwargs.enable_prefix_caching=false',
        'model_config.model_id=Qwen/Qwen3-1.7B',
        # dtype fix (P2): demo casts the ACTOR to fp32 (ref stays bf16);
        # bf16 params swallow 2e-7 Adam steps (ulp ~6e-5 >> lr) => frozen.
        'model_config.load_dtype="float32"',
        'reference_model_config.load_dtype="bfloat16"',
        'model_config.dtype="bfloat16"',
        'model_config.flash_attention_block_size=256',
    ):
      self.assertIn(needle, self.yaml_text, msg=f'missing override: {needle}')

  def test_vllm_sizes_match_demo_formula(self):
    # demo: max_num_seqs = prompts_per_step * NUM_GENERATIONS = 64 * 8;
    # max_batched_tokens = seqs * kv_cache // 8, kv = 1024 + 1024 + 256
    # (CLI computes kv_cache with the same formula, base_rl_pipeline.py:310).
    num_generations = 8
    batch_size = 64
    kv_cache = 1024 + 1024 + 256
    self.assertIn(f'vllm_config.max_num_seqs={batch_size * num_generations}',
                  self.yaml_text)
    self.assertIn(
        'vllm_config.max_num_batched_tokens='
        f'{batch_size * num_generations * kv_cache // 8}',
        self.yaml_text,
    )

  def test_rope_patch_matches_stream_yaml(self):
    def _patch_block(text):
      lines = []
      in_block = False
      for line in text.splitlines():
        stripped = line.rstrip()
        if '--- Hot Patch tpu-inference' in stripped:
          in_block = True
        if in_block:
          lines.append(stripped)
          if stripped.endswith('------------------------------------------'):
            break
      return lines

    aligned = _patch_block(self.yaml_text)
    stream = _patch_block(_read(_STREAM_YAML))
    self.assertNotEmpty(aligned)
    self.assertEqual(aligned, stream)


if __name__ == '__main__':
  absltest.main()
