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

"""Safetensors dtype codes shared by the loader and saver."""

from collections.abc import Mapping

import ml_dtypes
import numpy as np

SAFETENSORS_DTYPES: Mapping[str, np.dtype] = {
    'BOOL': np.dtype(np.bool_),
    'U8': np.dtype(np.uint8),
    'I8': np.dtype(np.int8),
    'U16': np.dtype(np.uint16),
    'I16': np.dtype(np.int16),
    'U32': np.dtype(np.uint32),
    'I32': np.dtype(np.int32),
    'U64': np.dtype(np.uint64),
    'I64': np.dtype(np.int64),
    'F8_E4M3': np.dtype(ml_dtypes.float8_e4m3fn),
    'F8_E5M2': np.dtype(ml_dtypes.float8_e5m2),
    'BF16': np.dtype(ml_dtypes.bfloat16),
    'F16': np.dtype(np.float16),
    'F32': np.dtype(np.float32),
    'F64': np.dtype(np.float64),
}
NP_TO_SAFETENSORS_DTYPE: Mapping[np.dtype, str] = {
    dt: name for name, dt in SAFETENSORS_DTYPES.items()
}
