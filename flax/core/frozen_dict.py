# Lint as: python3
# Copyright 2020 The Flax Authors.
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
"""Frozen dict.
"""

from typing import TypeVar, Mapping, Dict

import jax


K = TypeVar('K')
V = TypeVar('V')


class FrozenDict(Mapping[K, V]):
  """An immutable variant of dictionaries.
  """

  def __init__(self, *args, **kwargs):
    self._dict = dict(*args, **kwargs)
    self._hash = None

  def __getitem__(self, key):
    return self._dict[key]

  def __setitem__(self, key, value):
    raise ValueError('FrozenDict is immutable.')

  def __contains__(self, key):
    return key in self._dict

  def __iter__(self):
    return iter(self._dict)

  def __len__(self):
    return len(self._dict)

  def __repr__(self):
    return 'FrozenDict(%r)' % self._dict

  def __hash__(self):
    if self._hash is None:
      h = 0
      for key, value in self._dict.items():
        h ^= hash((key, value))
      self._hash = h
    return self._hash

  def copy(self, **add_or_replace):
    return type(self)(self, **add_or_replace)

  def items(self):
    return self._dict.items()

jax.tree_util.register_pytree_node(
    FrozenDict,
    lambda x: ((dict(x),), ()),
    lambda _, data: FrozenDict(data[0]))


def freeze(x: Dict[K, V]) -> FrozenDict[K, V]:
  """Freeze a nested dict."""
  if not isinstance(x, dict):
    return x
  temp = {}
  for key, value in x.items():
    temp[key] = freeze(value)
  return FrozenDict(temp)


def unfreeze(x: FrozenDict[K, V]) -> Dict[K, V]:
  if not isinstance(x, FrozenDict):
    return x
  temp = {}
  for key, value in x.items():
    temp[key] = unfreeze(value)
  return temp
