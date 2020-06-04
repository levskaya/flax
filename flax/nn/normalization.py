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

"""Normalization modules for Flax."""

from . import base

from jax import lax
from jax.nn import initializers
import jax.numpy as jnp
from ..core import Scope


_no_init = lambda rng, shape: ()


def _absolute_dims(rank, dims):
  return tuple([rank + dim if dim < 0 else dim for dim in dims])


def batch_norm(scope,
               x,
               use_running_average=False,
               axis=-1,
               momentum=0.99,
               epsilon=1e-5,
               dtype=jnp.float32,
               bias=True,
               scale=True,
               bias_init=initializers.zeros,
               scale_init=initializers.ones,
               axis_name=None,
               axis_index_groups=None):
  """Normalizes the input using batch statistics.

  Args:
    x: the input to be normalized.
    use_running_average: if true, the statistics stored in batch_stats
      will be used instead of computing the batch statistics on the input.
    axis: the feature or non-batch axis of the input.
    momentum: decay rate for the exponential moving average of
      the batch statistics.
    epsilon: a small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    bias:  if True, bias (beta) is added.
    scale: if True, multiply by scale (gamma).
      When the next layer is linear (also e.g. nn.relu), this can be disabled
      since the scaling will be done by the next layer.
    bias_init: initializer for bias, by default, zero.
    scale_init: initializer for scale, by default, one.
    axis_name: the axis name used to combine batch statistics from multiple
      devices. See `jax.pmap` for a description of axis names (default: None).
    axis_index_groups: groups of axis indices within that named axis
      representing subsets of devices to reduce over (default: None). For example,
      `[[0, 1], [2, 3]]` would independently batch-normalize over the examples
      on the first two and last two devices. See `jax.lax.psum` for more details.

  Returns:
    Normalized inputs (the same shape as inputs).
  """
  x = jnp.asarray(x, jnp.float32)
  axis = axis if isinstance(axis, tuple) else (axis,)
  axis = _absolute_dims(x.ndim, axis)
  feature_shape = tuple(d if i in axis else 1 for i, d in enumerate(x.shape))
  reduced_feature_shape = tuple(d for i, d in enumerate(x.shape) if i in axis)
  reduction_axis = tuple(i for i in range(x.ndim) if i not in axis)
  initializing = not scope.has_variable('stats', 'ra_mean')

  if initializing:
    ra_mean = initializers.zeros(None, reduced_feature_shape)
    ra_var = initializers.ones(None, reduced_feature_shape)
  else:
    ra_mean = scope.get_variable('stats', 'ra_mean')
    ra_var = scope.get_variable('stats', 'ra_var')

  if use_running_average:
    mean, var = ra_mean, ra_var
  else:
    mean = jnp.mean(x, axis=reduction_axis, keepdims=False)
    if axis_name is not None and not self.is_initializing():
      mean = lax.pmean(
          mean, axis_name=axis_name, axis_index_groups=axis_index_groups)

    mean2 = jnp.mean(lax.square(x), axis=reduction_axis, keepdims=False)
    if axis_name is not None and not self.is_initializing():
      mean2 = lax.pmean(
          mean2, axis_name=axis_name, axis_index_groups=axis_index_groups)
    var = mean2 - lax.square(mean)

    if not initializing:
      ra_mean = momentum * ra_mean + (1 - momentum) * mean
      ra_var = momentum * ra_var + (1 - momentum) * var

  scope.put_variable('stats', 'ra_mean', ra_mean)
  scope.put_variable('stats', 'ra_var', ra_var)

  y = x - mean.reshape(feature_shape)
  mul = lax.rsqrt(var + epsilon)
  if scale:
    mul = mul * scope.param(
        'scale', scale_init, reduced_feature_shape).reshape(feature_shape)
  y = y * mul
  if bias:
    y = y + scope.param(
        'bias', bias_init, reduced_feature_shape).reshape(feature_shape)
  return jnp.asarray(y, dtype)


def layer_norm(
    scope: Scope,
    x,
    epsilon=1e-6,
    dtype=jnp.float32,
    bias=True,
    scale=True,
    bias_init=initializers.zeros,
    scale_init=initializers.ones):
  """Applies layer normalization on the input.

  It normalizes the activations of the layer for each given example in a
  batch independently, rather than across a batch like Batch Normalization.
  i.e. applies a transformation that maintains the mean activation within
  each example close to 0 and the activation standard deviation close to 1.

  Args:
    x: the inputs
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    bias:  If True, bias (beta) is added.
    scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.

  Returns:
    Normalized inputs (the same shape as inputs).

  """
  features = x.shape[-1]
  mean = jnp.mean(x, axis=-1, keepdims=True)
  mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
  var = mean2 - lax.square(mean)
  mul = lax.rsqrt(var + epsilon)
  if scale:
    mul = mul * jnp.asarray(scope.param('scale', scale_init, (features,)),
                            dtype)
  y = (x - mean) * mul
  if bias:
    y = y + jnp.asarray(scope.param('bias', bias_init, (features,)), dtype)
  return y


def group_norm(scope,
               x,
               num_groups=32,
               group_size=None,
               epsilon=1e-6,
               dtype=jnp.float32,
               bias=True,
               scale=True,
               bias_init=initializers.zeros,
               scale_init=initializers.ones):
  """Applies group normalization to the input (arxiv.org/abs/1803.08494).

  This op is similar to batch normalization, but statistics are shared across
  equally-sized groups of channels and not shared across batch dimension.
  Thus, group normalization does not depend on the batch composition and does
  not require maintaining internal state for storing statistics.

  The user should either specify the total number of channel groups or the
  number of channels per group.

  Args:
    x: the input of shape N...C, where N is a batch dimension and C is a
      channels dimensions. `...` represents an arbitrary number of extra
      dimensions that are used to accumulate statistics over.
    num_groups: the total number of channel groups. The default value of 32 is
      proposed by the original group normalization paper.
    group_size: the number of channels in a group.
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: the dtype of the computation (default: float32).
    bias:  If True, bias (beta) is added.
    scale: If True, multiply by scale (gamma). When the next layer is linear
      (also e.g. nn.relu), this can be disabled since the scaling will be done
      by the next layer.
    bias_init: Initializer for bias, by default, zero.
    scale_init: Initializer for scale, by default, one.

  Returns:
    Normalized inputs (the same shape as inputs).

  """
  x = jnp.asarray(x, jnp.float32)
  if ((num_groups is None and group_size is None) or
      (num_groups is not None and group_size is not None)):
    raise ValueError('Either `num_groups` or `group_size` should be '
                     'specified, but not both of them.')

  if group_size is not None:
    channels = x.shape[-1]
    if channels % group_size != 0:
      raise ValueError('Number of channels ({}) is not multiple of the '
                       'group size ({}).'.format(channels, group_size))
    num_groups = channels // group_size

  input_shape = x.shape
  group_shape = x.shape[:-1] + (num_groups, x.shape[-1] // num_groups)

  x = x.reshape(group_shape)

  reduction_axis = list(range(1, x.ndim - 2)) + [x.ndim - 1]

  mean = jnp.mean(x, axis=reduction_axis, keepdims=True)
  mean_of_squares = jnp.mean(jnp.square(x), axis=reduction_axis,
                             keepdims=True)
  var = mean_of_squares - jnp.square(mean)

  x = (x - mean) * lax.rsqrt(var + epsilon)

  x = x.reshape(input_shape)

  feature_shape = tuple([1 for d in input_shape[:-1]] + [input_shape[-1]])
  if scale:
    x = x * scope.param('scale', scale_init, feature_shape)
  if bias:
    x = x + scope.param('bias', bias_init, feature_shape)

  return x.astype(dtype)
