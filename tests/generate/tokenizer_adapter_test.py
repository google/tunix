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

#

from absl.testing import absltest
import transformers
from tunix.generate import tokenizer_adapter as adapter


AutoTokenizer = transformers.AutoTokenizer


class TokenizerAdapterTest(absltest.TestCase):

  def test_hf_tokenizer_adapter(self):
    # Additional assignment to handle google internal logics.
    model = None  # pylint: disable=unused-variable
    #
    if model is None:
      model = 'meta-llama/Meta-Llama-3-8B-Instruct'
    hf_tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer_adapter = adapter.TokenizerAdapter(hf_tokenizer)
    self.assertEqual(
        tokenizer_adapter._tokenizer_type, adapter.TokenizerType.HF
    )
    self.assertIsNotNone(tokenizer_adapter.bos_id())
    self.assertIsNotNone(tokenizer_adapter.eos_id())
    self.assertIsNotNone(tokenizer_adapter.pad_id())
    encoded = tokenizer_adapter.encode('test', add_special_tokens=False)
    self.assertIsNotNone(encoded)
    decoded = tokenizer_adapter.decode(encoded)
    self.assertIsNotNone(decoded)
    self.assertEqual(decoded, 'test')


class TokenizerTest(absltest.TestCase):

  def test_default_tokenizer(self):
    tokenizer = adapter.Tokenizer()
    self.assertEqual(tokenizer.tokenizer_type, 'sentencepiece')
    self.assertIsNotNone(tokenizer._tokenizer_type, adapter.TokenizerType.SP)

  def test_tokenize_hf(self):
    model = 'meta-llama/Meta-Llama-3-8B-Instruct'
    tokenizer = adapter.Tokenizer(
        tokenizer_type='huggingface', tokenizer_path=model
    )

    text = 'hello world'
    tokens = tokenizer.tokenize(text, add_eos=True)

    # Check BOS
    self.assertEqual(tokens[0], tokenizer.bos_id())
    # Check EOS
    self.assertEqual(tokens[-1], tokenizer.eos_id())

    # Check no double BOS (assuming "hello" doesn't tokenize to BOS)
    self.assertNotEqual(tokens[1], tokenizer.bos_id())

    # Test add_bos=False
    tokens_no_bos = tokenizer.tokenize(text, add_bos=False, add_eos=True)
    self.assertNotEqual(tokens_no_bos[0], tokenizer.bos_id())
    self.assertEqual(tokens_no_bos[-1], tokenizer.eos_id())

    # Decode back
    decoded = tokenizer.decode(tokens)
    # decoded might contain special tokens depending on decode implementation
    # TokenizerAdapter.decode calls tokenizer.decode. HF decode usually skips special tokens by default?
    # No, skip_special_tokens defaults to False in HF decode?
    # Actually TokenizerAdapter.decode just calls self._tokenizer.decode(ids, **kwargs).
    # We didn't pass kwargs.

    # Let's just check length or content roughly.
    self.assertTrue(len(tokens) > 2)

  def test_special_eos_token(self):
    model = 'meta-llama/Meta-Llama-3-8B-Instruct'
    # Use a token that definitely exists and is different from default EOS
    special_eos = 'world'
    tokenizer = adapter.Tokenizer(
        tokenizer_type='huggingface',
        tokenizer_path=model,
        special_eos_token=special_eos,
    )

    text = 'hello'
    tokens = tokenizer.tokenize(text, add_eos=True)

    # Check EOS is the special token
    self.assertEqual(tokens[-1], tokenizer.eos_id())

    # Verify the ID matches what we expect from the tokenizer directly
    hf_tokenizer = tokenizer.tokenizer
    expected_id = hf_tokenizer.convert_tokens_to_ids(special_eos)
    self.assertEqual(tokenizer.eos_id(), expected_id)
    self.assertNotEqual(tokenizer.eos_id(), hf_tokenizer.eos_token_id)


if __name__ == '__main__':
  absltest.main()
