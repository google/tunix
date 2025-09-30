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


if __name__ == '__main__':
  absltest.main()
