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

"""Compatibility re-export for MaxText until MaxText migrates to tunix.dist_rl."""

# TODO(tsbao): Remove this shim once MaxText migrates to tunix.dist_rl.
# Both `from ... import *` and `sys.modules[__name__] = _abstract_trainer` are
# intentional: static type checkers (pytype) inspect the wildcard import, while
# the `sys.modules` alias preserves runtime module identity and `mock.patch`
# compatibility across `tunix.experimental` and `tunix.dist_rl`.
import sys
from tunix.dist_rl.train import abstract_trainer as _abstract_trainer
from tunix.dist_rl.train.abstract_trainer import *  # pylint: disable=wildcard-import,unused-wildcard-import

sys.modules[__name__] = _abstract_trainer
