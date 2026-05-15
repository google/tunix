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

"""Utilities for enforcing immutability in Python objects."""

import abc


class ImmutableMeta(abc.ABCMeta):
  """Metaclass that sets `self._locked = True` after `__init__` completes.

  By overriding `__call__`, this metaclass ensures that the `_locked` flag is
  only set after the entire instantiation process (including the full
  `__init__` chain of subclasses) has finished. This allows subclasses to
  perform their own initialization normally before the immutability constraint
  is applied.

  Note that this metaclass alone does not enforce immutability; it only provides
  the signal. Classes must explicitly implement `__setattr__` to check the
  `_locked` flag and raise an error (as shown in the usage example).

  Usage:
    class MyImmutableClass(metaclass=ImmutableMeta):

      def __setattr__(self, name, value):
        if getattr(self, "_locked", False):
          raise AttributeError("Immutable!")
        super().__setattr__(name, value)
  """

  def __call__(cls, *args, **kwargs):
    """Create and initialize a new instance, then lock it against modification."""
    # This calls __new__ and __init__, completing the full initialization chain
    instance = super().__call__(*args, **kwargs)
    # Lock the instance after init is done
    object.__setattr__(instance, "_locked", True)
    return instance
