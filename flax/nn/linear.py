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

# Lint as: python3
"""Linear modules."""

from collections.abc import Iterable  # pylint: disable=g-importing-member
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Any

from . import base
from . import initializers

from jax import lax

import jax.numpy as jnp
import numpy as onp


default_kernel_init = initializers.lecun_normal()


def _normalize_axes(axes, ndim):
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


class DenseGeneral(base.Module):
  """A linear transformation with flexible axes.

  Args:
    features: tuple with numbers of output features.
    axis: tuple with axes to apply the transformation on.
    batch_dims: tuple with batch axes.
    bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
  """
  def __init__(self,
               features: int,
               axis: int = -1,
               batch_dims: Sequence[int] = (),
               bias: bool = True,
               dtype: Any = jnp.float32,
               kernel_init: Callable = default_kernel_init,
               bias_init: Callable = initializers.zeros,
               precision: Any = None,
               name: Optional[str] = None):
    self.features = features
    self.axis = axis
    self.batch_dims = batch_dims
    self.bias = bias
    self.dtype = dtype
    self.kernel_init = kernel_init
    self.bias_init = bias_init
    self.precision = precision
    if not isinstance(self.features, Iterable):
      self.features = (self.features,)
    if not isinstance(self.axis, Iterable):
      self.axis = (self.axis,)
    if not isinstance(self.batch_dims, Iterable):
      self.batch_dims = (self.batch_dims,)
    self.features = tuple(self.features)
    self.axis = tuple(self.axis)
    self.batch_dims = tuple(self.batch_dims)
    if self.batch_dims:
      max_dim = onp.max(self.batch_dims)
      if set(self.batch_dims) != set(range(max_dim + 1)):
        raise ValueError('batch_dims %s must be consecutive leading '
                         'dimensions starting from 0.' % str(batch_dims))
    self.name = name

  def apply(self, inputs):
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.
    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    ndim = inputs.ndim
    n_batch_dims = len(self.batch_dims)
    axis = _normalize_axes(self.axis, ndim)
    batch_dims = _normalize_axes(self.batch_dims, ndim)
    n_axis, n_features = len(axis), len(self.features)

    def kernel_init_wrap(rng, shape, dtype=jnp.float32):
      size_batch_dims = onp.prod(shape[:n_batch_dims], dtype=onp.int32)
      flat_shape = (onp.prod(shape[n_batch_dims:n_axis + n_batch_dims]),
                    onp.prod(shape[-n_features:]),)
      kernel = jnp.concatenate([self.kernel_init(rng, flat_shape, self.dtype)
                                for _ in range(size_batch_dims)], axis=0)
      return jnp.reshape(kernel, shape)

    batch_shape = tuple([inputs.shape[ax] for ax in batch_dims])
    kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + self.features
    kernel = self.param('kernel', batch_shape + kernel_shape, kernel_init_wrap)
    kernel = jnp.asarray(kernel, self.dtype)

    batch_ind = tuple(range(n_batch_dims))
    contract_ind = tuple(range(n_batch_dims, n_axis + n_batch_dims))
    out = lax.dot_general(inputs,
                          kernel,
                          ((axis, contract_ind), (batch_dims, batch_ind)),
                          precision=self.precision)
    if self.bias:
      def bias_init_wrap(rng, shape, dtype=jnp.float32):
        size_batch_dims = onp.prod(shape[:n_batch_dims], dtype=onp.int32)
        flat_shape = (onp.prod(shape[-n_features:]),)
        bias = jnp.concatenate([self.bias_init(rng, flat_shape, dtype)
                                for _ in range(size_batch_dims)], axis=0)
        return jnp.reshape(bias, shape)

      bias = self.param('bias', batch_shape + self.features, bias_init_wrap)

      # Reshape bias for broadcast.
      expand_dims = sorted(
          set(range(inputs.ndim)) - set(axis) - set(batch_dims))
      for ax in expand_dims:
        bias = jnp.expand_dims(bias, ax)
      bias = jnp.asarray(bias, self.dtype)
      out = out + bias
    return out


@dataclass
class Dense(base.Module):
  """A linear transformation applied over the last dimension of the input."""
  features: int
  bias: bool = True
  dtype: Any = jnp.float32
  precision: Any = None
  kernel_init: Any = default_kernel_init
  bias_init: Any = initializers.zeros
  name: Optional[str] = None

  def apply(self, inputs):
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.
      features: the number of output features.
      bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    kernel = self.param('kernel', (inputs.shape[-1], self.features), self.kernel_init)
    kernel = jnp.asarray(kernel, self.dtype)
    y = lax.dot_general(inputs, kernel,
                        (((inputs.ndim - 1,), (0,)), ((), ())),
                        precision=self.precision)
    if self.bias:
      bias = self.param('bias', (self.features,), self.bias_init)
      bias = jnp.asarray(bias, self.dtype)
      y = y + bias
    return y


def _conv_dimension_numbers(input_shape):
  """Computes the dimension numbers based on the input shape."""
  ndim = len(input_shape)
  lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
  rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
  out_spec = lhs_spec
  return lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)


class Conv(base.Module):
  """Convolution Module wrapping lax.conv_general_dilated."""

  def apply(self,
            inputs,
            features,
            kernel_size,
            strides=None,
            padding='SAME',
            input_dilation=None,
            kernel_dilation=None,
            feature_group_count=1,
            bias=True,
            dtype=jnp.float32,
            precision=None,
            kernel_init=default_kernel_init,
            bias_init=initializers.zeros):
    """Applies a convolution to the inputs.

    Args:
      inputs: input data with dimensions (batch, spatial_dims..., features).
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel.
      strides: a sequence of `n` integers, representing the inter-window
        strides.
      padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
        of `n` `(low, high)` integer pairs that give the padding to apply before
        and after each spatial dimension.
      input_dilation: `None`, or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of `inputs`.
        Convolution with input dilation `d` is equivalent to transposed
        convolution with stride `d`.
      kernel_dilation: `None`, or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel. Convolution with kernel dilation is also known as 'atrous
        convolution'.
      feature_group_count: integer, default 1. If specified divides the input
        features into groups.
      bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
    Returns:
      The convolved data.
    """

    inputs = jnp.asarray(inputs, dtype)

    if strides is None:
      strides = (1,) * (inputs.ndim - 2)

    in_features = inputs.shape[-1]
    assert in_features % feature_group_count == 0
    kernel_shape = kernel_size + (in_features // feature_group_count, features)
    kernel = self.param('kernel', kernel_shape, kernel_init)
    kernel = jnp.asarray(kernel, dtype)

    dimension_numbers = _conv_dimension_numbers(inputs.shape)
    y = lax.conv_general_dilated(
        inputs,
        kernel,
        strides,
        padding,
        lhs_dilation=input_dilation,
        rhs_dilation=kernel_dilation,
        dimension_numbers=dimension_numbers,
        feature_group_count=feature_group_count,
        precision=precision)

    if bias:
      bias = self.param('bias', (features,), bias_init)
      bias = jnp.asarray(bias, dtype)
      y = y + bias
    return y


class ConvTranspose(base.Module):
  """Transposed convolution Module wrapping lax.conv_transpose."""

  def apply(self,
            inputs,
            features,
            kernel_size,
            strides=None,
            padding='SAME',
            kernel_dilation=None,
            bias=True,
            dtype=jnp.float32,
            precision=None,
            kernel_init=default_kernel_init,
            bias_init=initializers.zeros):
    """Applies a transposed convolution to the inputs. Behaviour mirrors that of
    `jax.lax.conv_transpose`.

    Args:
      inputs: input data with dimensions (batch, spatial_dims..., features).
      features: number of convolution filters.
      kernel_size: shape of the convolutional kernel.
      strides: a sequence of `n` integers, representing the inter-window
        strides.
      padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
        of `n` `(low, high)` integer pairs that give the padding to apply before
        and after each spatial dimension.
      kernel_dilation: `None`, or a sequence of `n` integers, giving the
        dilation factor to apply in each spatial dimension of the convolution
        kernel. Convolution with kernel dilation is also known as 'atrous
        convolution'.
      bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: float32).
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the convolutional kernel.
      bias_init: initializer for the bias.
    Returns:
      The convolved data.
    """
    inputs = jnp.asarray(inputs, dtype)
    strides = strides or (1,) * (inputs.ndim - 2)

    in_features = inputs.shape[-1]
    kernel_shape = kernel_size + (in_features, features)
    kernel = self.param('kernel', kernel_shape, kernel_init)
    kernel = jnp.asarray(kernel, dtype)

    y = lax.conv_transpose(inputs, kernel, strides, padding,
                           rhs_dilation=kernel_dilation, precision=precision)

    if bias:
      bias = self.param('bias', (features,), bias_init)
      bias = jnp.asarray(bias, dtype)
      y = y + bias
    return y


default_embed_init = initializers.variance_scaling(1.0, 'fan_in', 'normal',
                                                   out_axis=0)


@dataclass
class Embed(base.Module):
  """Embedding Module.

  A parameterized function from integers [0, n) to d-dimensional vectors.
  """
  num_embeddings: int
  features: int
  embedding_init: Callable = default_embed_init
  name: Optional[str] = None

  def apply(self, inputs):
    """Embeds the inputs along the last dimension.

    Args:
      inputs: input data, all dimensions are considered batch dimensions.
      num_embeddings: number of embeddings.
      features: Number of feature dimensions for each embedding.
      embedding_init: embedding initializer.

    Returns:
      Output which is embedded input data.  The output shape follows the input,
      with an additional `features` dimension appended.
    """
    if inputs.dtype not in [jnp.int32, jnp.int64, jnp.uint32, jnp.uint64]:
      raise ValueError('Input type must be an integer or unsigned integer.')
    embedding = self.param('embedding', (self.num_embeddings, self.features),
                           self.embedding_init)
    return embedding[inputs]

  def attend(self, query, **unused_kwargs):
    """Attend over the embedding using a query array.

    Args:
      query: array with last dimension equal the feature depth `features` of the
        embedding.
      **unused_kwargs: unused arguments passed from the apply method.

    Returns:
      An array with final dim `num_embeddings` corresponding to the batched
      inner-product of the array of query vectors against each embedding.
      Commonly used for weight-sharing between embeddings and logit transform
      in NLP models.
    """
    del unused_kwargs
    embedding = self.get_param('embedding')
    return lax.dot_general(
        query, embedding, (((query.ndim - 1,), (1,)), ((), ())))
