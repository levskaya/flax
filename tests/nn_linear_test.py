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

"""Tests for flax.nn.linear."""

import functools

from absl.testing import absltest
from absl.testing import parameterized

from flax.core import init, apply
from flax import nn

import jax
from jax import random
from jax.nn import initializers
import jax.numpy as jnp

import numpy as onp

# Parse absl flags test_srcdir and test_tmpdir.
jax.config.parse_flags_with_absl()


class LinearTest(parameterized.TestCase):

  def test_dense(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))
    dense_module = functools.partial(
        nn.dense,
        features=4,
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = init(dense_module)(rng, x)
    self.assertEqual(y.shape, (1, 4))
    onp.testing.assert_allclose(y, onp.full((1, 4), 4.))

  def test_dense_extra_batch_dims(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 2, 3))
    dense_module = functools.partial(
        nn.dense,
        features=4,
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = init(dense_module)(rng, x)
    onp.testing.assert_allclose(y, onp.full((1, 2, 4), 4.))

  def test_dense_no_bias(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))
    dense_module = functools.partial(
        nn.dense,
        features=4,
        bias=False,
        kernel_init=initializers.ones,
    )
    y, _ = init(dense_module)(rng, x)
    onp.testing.assert_allclose(y, onp.full((1, 4), 3.))

  def test_dense_is_dense_general(self):
    x = jax.random.normal(random.PRNGKey(0), (5, 3))
    dense_module = functools.partial(nn.dense,
        features=4,
        bias=True,
        bias_init=initializers.normal(),
    )
    y1, _ = init(dense_module)(random.PRNGKey(1), x)
    dg_module = functools.partial(
        nn.dense_general,
        features=4,
        bias=True,
        bias_init=initializers.normal(),
    )
    y2, _ = init(dg_module)(random.PRNGKey(1), x)

    onp.testing.assert_allclose(y1, y2)

  def test_dense_general_batch_dim_raises(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3, 2, 5))
    with self.assertRaises(ValueError):
      dg_module = functools.partial(
          nn.dense_general,
          features=4,
          batch_dims=(0, 2),
          kernel_init=initializers.ones,
          bias_init=initializers.ones,
      )
      init(dg_module)(rng, x)

  def test_dense_general_two_out(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 3))
    dg_module = functools.partial(
        nn.dense_general,
        features=(2, 2),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = init(dg_module)(rng, x)
    onp.testing.assert_allclose(y, onp.full((1, 2, 2), 4.))

  def test_dense_general_two_in(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 2, 2))
    dg_module = functools.partial(
        nn.dense_general,
        features=3,
        axis=(-2, 2),
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, _ = init(dg_module)(rng, x)
    onp.testing.assert_allclose(y, onp.full((1, 3), 5.))

  def test_dense_general_batch_dim(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((2, 1, 3, 5))

    state = {'counter': 0.}
    def _counter_init(rng, shape, dtype, state):
      del rng, dtype
      state['counter'] += 1.
      return jnp.full(shape, state['counter'])
    counter_init = functools.partial(_counter_init, state=state)

    dg_module = functools.partial(
        nn.dense_general,
        features=7,
        axis=(3, -2),
        batch_dims=0,
        bias_init=initializers.ones,
        kernel_init=counter_init,
    )
    y, _ = init(dg_module)(rng, x)
    target = onp.concatenate(
        [onp.full((1, 1, 7), 16.), onp.full((1, 1, 7), 31.)], axis=0)
    onp.testing.assert_allclose(y, target)

  @parameterized.parameters([((-2, 3), (), 'bijk,jklm->bilm'),
                             ((3, -2), (), 'bijk,kjlm->bilm'),
                             ((-2, 3), (0,), 'bijk,bjklm->bilm')])
  def test_dense_general_vs_numpy(self, axis, batch_dims, einsum_expr):
    rng = random.PRNGKey(0)
    x = jnp.ones((16, 8, 9, 10))

    dg_module = functools.partial(
        nn.dense_general,
        features=(11, 12),
        axis=axis,
        batch_dims=batch_dims,
        bias_init=initializers.ones,
        kernel_init=initializers.normal(),
    )
    y, initial_params = init(dg_module)(rng, x)
    target = onp.einsum(einsum_expr, x, initial_params['param']['kernel']) + 1.
    onp.testing.assert_allclose(y, target, atol=1e-6)

  def test_conv(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 8, 3))
    conv_module = functools.partial(nn.conv,
        features=4,
        kernel_size=(3,),
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, initial_params = init(conv_module)(rng, x)
    self.assertEqual(initial_params['param']['kernel'].shape, (3, 3, 4))
    onp.testing.assert_allclose(y, onp.full((1, 6, 4), 10.))

  def test_group_conv(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 8, 4))
    conv_module = functools.partial(nn.conv,
        features=4,
        kernel_size=(3,),
        feature_group_count=2,
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, initial_params = init(conv_module)(rng, x)
    self.assertEqual(initial_params['param']['kernel'].shape, (3, 2, 4))
    onp.testing.assert_allclose(y, onp.full((1, 6, 4), 7.))

  def test_conv_transpose(self):
    rng = random.PRNGKey(0)
    x = jnp.ones((1, 8, 3))
    conv_transpose_module = functools.partial(nn.conv_transpose,
        features=4,
        kernel_size=(3,),
        padding='VALID',
        kernel_init=initializers.ones,
        bias_init=initializers.ones,
    )
    y, initial_params = init(conv_transpose_module)(rng, x)
    self.assertEqual(initial_params['param']['kernel'].shape, (3, 3, 4))
    correct_ans = onp.array([[[ 4.,  4.,  4.,  4.],
                              [ 7.,  7.,  7.,  7.],
                              [10., 10., 10., 10.],
                              [10., 10., 10., 10.],
                              [10., 10., 10., 10.],
                              [10., 10., 10., 10.],
                              [10., 10., 10., 10.],
                              [10., 10., 10., 10.],
                              [ 7.,  7.,  7.,  7.],
                              [ 4.,  4.,  4.,  4.]]])
    onp.testing.assert_allclose(y, correct_ans)

  def test_embed(self):
    rng = random.PRNGKey(0)
    x = jnp.arange(4)[None]
    dummy_embedding = jnp.broadcast_to(
        jnp.arange(4)[..., None], (4, 3)).astype(jnp.float32)
    embed_module = functools.partial(
        nn.embedding,
        num_embeddings=4,
        features=3,
        init_fn=lambda rng, shape: dummy_embedding,
    )
    lookup_fn = lambda scope, x: embed_module(scope).lookup(x)
    attend_fn = lambda scope, x: embed_module(scope).attend(x)
    y, initial_params = init(lookup_fn)(rng, x)
    onp.testing.assert_allclose(y, dummy_embedding[None])

    z = apply(attend_fn)(initial_params, jnp.ones((3,)))
    onp.testing.assert_allclose(z, 3. * jnp.arange(4))


if __name__ == '__main__':
  absltest.main()
