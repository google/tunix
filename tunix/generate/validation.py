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

"""Some validation functions for the sampler."""


def check_sampling_mode_conflict(
    original_sampling_mode: list[
        str | None
    ],  # pass in as list to modify in place
    new_sampling_mode: str,
) -> None:
  """Checks if the new sampling mode conflicts with the original sampling mode."""

  if original_sampling_mode[0] is not None:
    raise ValueError(
        'Conflicts setting sampling_mode, the current set sampling_mode is'
        f' {original_sampling_mode[0]} but trying to override to'
        f' {new_sampling_mode}. The rules are\n: 1. If top_p is provided,'
        ' top_p will be used. 2. If top_k > 1 and penalty_alpha > 0,'
        ' contrastive_search will be used. 3. If beam_size is provided,'
        ' beam_search will be used 4. If none of the above, greedy will be'
        ' used.'
    )
  else:
    original_sampling_mode[0] = new_sampling_mode
