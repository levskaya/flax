from typing import Any, Callable, Optional, Sequence
from functools import partial

import jax
from jax import lax, random
from jax import numpy as jnp

from flax.core import Scope, init, apply, lift, Array, nn



def softmax_attn(scope: Scope, weights: Array):
  del scope
  norm_dims = tuple(range(weights.ndim // 2, weights.ndim))
  log_norms = jax.scipy.special.logsumexp(
      weights, axis=norm_dims, keepdims=True)
  return jnp.exp(weights - log_norms)

def with_dropout(fn, rate: float, deterministic: bool = False):
  def attn_fn(scope: Scope, weights: Array):
    attn_weights = fn(scope, weights)
    return nn.dropout(scope, attn_weights, deterministic=deterministic, rate=rate)
  return attn_fn

def _dot_product_attention(
    scope: Scope,
    query: Array, key: Array, value: Array,
    bias: Optional[Array] = None,
    attn_fn: Callable = softmax_attn,
    dtype=jnp.float32):
  assert key.ndim == query.ndim
  assert key.ndim == value.ndim

  n = query.ndim
  attn_weights = lax.dot_general(
      query, key,
      (((n-1,), (n - 1,)), ((), ())))
  if bias is not None:
    attn_weights += bias
  attn_weights = attn_fn(scope, attn_weights)
  attn_weights = attn_weights.astype(dtype)

  contract_dims = (
      tuple(range(n - 1, attn_weights.ndim)),
      tuple(range(0, n  - 1)))
  y = lax.dot_general(
      attn_weights, value,
      (contract_dims, ((), ())))
  return y


def dot_product_attention(
    scope: Scope,
    inputs_q: Array,
    inputs_kv: Array,
    bias: Optional[Array] = None,
    qkv_features: Optional[int] = None,
    out_features: Optional[int] = None,
    attn_fn: Callable = softmax_attn,
    dtype=jnp.float32):
  if qkv_features is None:
    qkv_features = inputs_q.shape[-1]
  if out_features is None:
    out_features = inputs_q.shape[-1]
  dense = partial(nn.dense, features=qkv_features, bias=False, dtype=dtype)

  query = scope.child(dense, 'query')(inputs_q)
  key = scope.child(dense, 'key')(inputs_kv)
  value = scope.child(dense, 'value')(inputs_kv)

  y = _dot_product_attention(
      scope, query, key, value,
      bias=bias,
      attn_fn=attn_fn, dtype=dtype)

  return scope.child(nn.dense, 'out')(y, features=out_features, dtype=dtype)



def multi_head_dot_product_attention(
    scope: Scope,
    inputs_q: Array,
    inputs_kv: Array,
    bias: Optional[Array] = None,
    qkv_features: Optional[int] = None,
    out_features: Optional[int] = None,
    attn_fn: Callable = softmax_attn,
    batch_axes: Sequence[int] = (0,),
    num_heads: int = 1,
    dtype=jnp.float32,
    broadcast_dropout=False):

  if qkv_features is None:
    qkv_features = inputs_q.shape[-1]
  if out_features is None:
    out_features = inputs_q.shape[-1]

  attn_fn = partial(
      dot_product_attention,
      attn_fn=attn_fn,
      qkv_features=qkv_features // num_heads,
      out_features=out_features,
      dtype=dtype)
  attn_fn = lift.vmap(
      attn_fn,
      in_axes=(None, None, None), out_axes=-2,
      axis_size=num_heads,
      variable_in_axes={'param': 0},
      variable_out_axes={'param': 0},
      split_rngs={'param': True, 'dropout': not broadcast_dropout})
  for axis in reversed(sorted(batch_axes)):
    attn_fn = lift.vmap(
        attn_fn,
        in_axes=(axis, axis, axis), out_axes=axis,
        variable_in_axes={'param': None},
        variable_out_axes={'param': None},
        split_rngs={'param': False, 'dropout': not broadcast_dropout})

  y = attn_fn(scope, inputs_q, inputs_kv, bias)
  return y.mean(axis=-2)

if __name__ == "__main__":
  inputs = jnp.ones((2, 7, 16))

  y, variables = init(multi_head_dot_product_attention)(
      {'param': random.PRNGKey(0), 'dropout': random.PRNGKey(1)},
      inputs, inputs,
      num_heads=2,
      batch_axes=(0,),
      attn_fn=with_dropout(softmax_attn, 0.1, deterministic=False)
      )

  print(y.shape)
  print(jax.tree_map(jnp.shape, variables))