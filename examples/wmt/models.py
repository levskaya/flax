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
# Copyright 2020 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer-based machine translation model."""

from typing import Callable, Any

from flax import nn
from flax import struct

from jax import lax
import jax.numpy as jnp
import numpy as np


def shift_right(x, axis=1):
  """Shift the input to the right by padding on axis 1."""
  pad_widths = [(0, 0)] * len(x.shape)
  pad_widths[axis] = (1, 0)
  padded = jnp.pad(
      x, pad_widths, mode='constant', constant_values=x.dtype.type(0))
  return padded[:, :-1]


def sinusoidal_init(max_len=2048,
                    min_scale=1.0,
                    max_scale=10000.0):
  """1D Sinusoidal Position Embedding Initializer.

  Args:
      max_len: maximum possible length for the input.
      min_scale: float: minimum frequency-scale in sine grating.
      max_scale: float: maximum frequency-scale in sine grating.

  Returns:
      output: init function returning `(1, max_len, d_feature)`
  """

  def init(key, shape, dtype=np.float32):
    """Sinusoidal init."""
    del key, dtype
    d_feature = shape[-1]
    pe = np.zeros((max_len, d_feature), dtype=np.float32)
    position = np.arange(0, max_len)[:, np.newaxis]
    scale_factor = -np.log(max_scale / min_scale) / (d_feature // 2 - 1)
    div_term = min_scale * np.exp(np.arange(0, d_feature // 2) * scale_factor)
    pe[:, :d_feature // 2] = np.sin(position * div_term)
    pe[:, d_feature // 2: 2 * (d_feature // 2)] = np.cos(position * div_term)
    pe = pe[np.newaxis, :, :]  # [1, max_len, d_feature]
    return jnp.array(pe)

  return init


def add_position_embs(
    scope,
    inputs,
    inputs_positions=None,
    max_len=512,
    posemb_init=None,
    cache=None):
  """Applies AddPositionEmbs module.

  By default this layer uses a fixed sinusoidal embedding table. If a
  learned position embedding is desired, pass an initializer to
  posemb_init.

  Args:
    inputs: input data.
    inputs_positions: input position indices for packed sequences.
    max_len: maximum possible length for the input.
    posemb_init: positional embedding initializer, if None, then use a
      fixed (non-learned) sinusoidal embedding table.
    cache: flax attention cache for fast decoding.

  Returns:
    output: `(bs, timesteps, in_dim)`
  """
  # inputs.shape is (batch_size, seq_len, emb_dim)
  assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                            ' but it is: %d' % inputs.ndim)
  length = inputs.shape[1]
  pos_emb_shape = (1, max_len, inputs.shape[-1])
  if posemb_init is None:
    # Use a fixed (non-learned) sinusoidal position embedding.
    pos_embedding = sinusoidal_init(
        max_len=max_len)(None, pos_emb_shape, None)
  else:
    pos_embedding = scope.param('pos_embedding', posemb_init, pos_emb_shape)
  pe = pos_embedding[:, :length, :]

  if cache:
    if not scope.has_variable('cache', 'idx'):
      cache_idx = jnp.zeros((), jnp.uint32)
    else:
      cache_idx = scope.get_variable('cache', 'idx')
      cache_idx = cache_idx + 1
      _, _, df = pos_embedding.shape
      pe = lax.dynamic_slice(pos_embedding,
                              jnp.array((0, cache_idx, 0)),
                              jnp.array((1, 1, df)))
    scope.put_variable('cache', 'idx', cache_idx)

  if inputs_positions is None:
    # normal unpacked case:
    return inputs + pe
  else:
    # for packed data we need to use known position indices:
    return inputs + jnp.take(pe[0], inputs_positions, axis=0)


def mlp_block(
    scope,
    inputs,
    mlp_dim,
    dtype=jnp.float32,
    out_dim=None,
    dropout_rate=0.1,
    deterministic=False,
    kernel_init=nn.initializers.xavier_uniform(),
    bias_init=nn.initializers.normal(stddev=1e-6)):
  """Applies Transformer MlpBlock module."""
  actual_out_dim = inputs.shape[-1] if out_dim is None else out_dim
  x = scope.child(nn.dense)(inputs, mlp_dim, dtype=dtype, kernel_init=kernel_init,
                            bias_init=bias_init)
  x = nn.relu(x)
  x = scope.child(nn.dropout)(x, rate=dropout_rate, deterministic=deterministic)
  output = scope.child(nn.dense)(x, actual_out_dim, dtype=dtype, kernel_init=kernel_init,
                                 bias_init=bias_init)
  output = scope.child(nn.dropout)(output, rate=dropout_rate, deterministic=deterministic)
  return output


def encoder_1d_block(
    scope,
    inputs,
    qkv_dim,
    mlp_dim,
    num_heads,
    dtype=jnp.float32,
    inputs_segmentation=None,
    padding_mask=None,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
    deterministic=False):
  """Applies Encoder1DBlock module.

  Args:
    inputs: input data.
    qkv_dim: dimension of the query/key/value.
    mlp_dim: dimension of the mlp on top of attention block.
    num_heads: number of heads.
    dtype: the dtype of the computation (default: float32).
    inputs_segmentation: input segmentation info for packed examples.
    padding_mask: bool, mask padding tokens.
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate for attention weights.
    deterministic: bool, deterministic or not (to apply dropout).

  Returns:
    output after transformer encoder block.
  """

  # Attention block.
  assert inputs.ndim == 3
  x = scope.child(nn.layer_norm)(inputs, dtype=dtype)
  x = scope.child(nn.multi_head_dot_product_attention)(
      x,
      num_heads=num_heads,
      dtype=dtype,
      inputs_kv=x,
      qkv_features=qkv_dim,
      attention_axis=(1,),
      causal_mask=False,
      segmentation=inputs_segmentation,
      padding_mask=padding_mask,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6),
      bias=False,
      broadcast_dropout=False,
      dropout_rate=attention_dropout_rate,
      deterministic=deterministic)
  x = scope.child(nn.dropout)(x, rate=dropout_rate, deterministic=deterministic)
  x = x + inputs

  # MLP block.
  y = scope.child(nn.layer_norm)(x, dtype=dtype)
  y = scope.child(mlp_block)(
      y,
      mlp_dim=mlp_dim,
      dtype=dtype,
      dropout_rate=dropout_rate,
      deterministic=deterministic)

  return x + y


def encoder_decoder_1d_block(
    scope,
    targets,
    encoded,
    qkv_dim,
    mlp_dim,
    num_heads,
    dtype=jnp.float32,
    inputs_segmentation=None,
    targets_segmentation=None,
    padding_mask=None,
    key_padding_mask=None,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
    deterministic=False,
    cache=None):
  """Applies EncoderDecoder1DBlock module.

  Args:
    targets: input data for decoder
    encoded: input data from encoder
    qkv_dim: dimension of the query/key/value
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: number of heads
    dtype: the dtype of the computation (default: float32)
    inputs_segmentation: input segmentation info for packed examples.
    targets_segmentation: target segmentation info for packed examples.
    causal_mask: bool, mask future or not
    padding_mask: bool, mask padding tokens
    key_padding_mask: bool, mask padding tokens
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights
    deterministic: bool, deterministic or not (to apply dropout)
    cache: flax attention cache for fast decoding.

  Returns:
    output after transformer encoder-decoder block.
  """

  # Decoder block.
  assert targets.ndim == 3
  x = scope.child(nn.layer_norm)(targets, dtype=dtype)
  x = scope.child(nn.multi_head_dot_product_attention)(
      x,
      num_heads=num_heads,
      dtype=dtype,
      inputs_kv=x,
      qkv_features=qkv_dim,
      attention_axis=(1,),
      causal_mask=True,
      padding_mask=padding_mask,
      segmentation=targets_segmentation,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6),
      bias=False,
      broadcast_dropout=False,
      dropout_rate=attention_dropout_rate,
      deterministic=deterministic,
      cache=cache)
  x = scope.child(nn.dropout)(x, rate=dropout_rate, deterministic=deterministic)
  x = x + targets

  # Encoder-Decoder block.
  y = scope.child(nn.layer_norm)(x, dtype=dtype)
  y = scope.child(nn.multi_head_dot_product_attention)(
      y,
      num_heads=num_heads,
      dtype=dtype,
      inputs_kv=encoded,
      qkv_features=qkv_dim,
      attention_axis=(1,),
      causal_mask=False,
      padding_mask=padding_mask,
      key_padding_mask=key_padding_mask,
      segmentation=targets_segmentation,
      key_segmentation=inputs_segmentation,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6),
      bias=False,
      broadcast_dropout=False,
      dropout_rate=attention_dropout_rate,
      deterministic=deterministic)
  y = scope.child(nn.dropout)(y, rate=dropout_rate, deterministic=deterministic)
  y = y + x

  # MLP block.
  z = scope.child(nn.layer_norm)(y, dtype=dtype)
  z = scope.child(mlp_block)(
      z,
      mlp_dim=mlp_dim,
      dtype=dtype,
      dropout_rate=dropout_rate,
      deterministic=deterministic)

  return y + z


def encoder(
    scope,
    inputs,
    vocab_size,
    inputs_positions=None,
    inputs_segmentation=None,
    shared_embedding=None,
    use_bfloat16=False,
    emb_dim=512,
    num_heads=8,
    num_layers=6,
    qkv_dim=512,
    mlp_dim=2048,
    max_len=512,
    train=True,
    dropout_rate=0.1,
    attention_dropout_rate=0.1):
  """Applies Transformer model on the inputs.

  Args:
    inputs: input data
    vocab_size: size of the vocabulary
    inputs_positions: input subsequence positions for packed examples.
    inputs_segmentation: input segmentation info for packed examples.
    shared_embedding: a shared embedding layer to use.
    use_bfloat16: bool: whether use bfloat16.
    emb_dim: dimension of embedding
    num_heads: number of heads
    num_layers: number of layers
    qkv_dim: dimension of the query/key/value
    mlp_dim: dimension of the mlp on top of attention block
    max_len: maximum length.
    train: if it is training,
    dropout_rate: dropout rate
    attention_dropout_rate: dropout rate for attention weights

  Returns:
    output of a transformer encoder.
  """
  assert inputs.ndim == 2  # (batch, len)

  # Padding Masks
  src_padding_mask = (inputs > 0)[..., None]

  # Input Embedding
  if shared_embedding is None:
    input_embed = scope.child(nn.embedding)(
        num_embeddings=vocab_size,
        features=emb_dim,
        init_fn=nn.initializers.normal(stddev=1.0))
  else:
    input_embed = shared_embedding
  x = inputs.astype('int32')
  x = input_embed.lookup(x)
  x = scope.child(add_position_embs, 'posembed_input')(
      x,
      inputs_positions=inputs_positions,
      max_len=max_len)
  x = scope.child(nn.dropout)(x, rate=dropout_rate, deterministic=not train)

  if use_bfloat16:
    x = x.astype(jnp.bfloat16)
    dtype = jnp.bfloat16
  else:
    dtype = jnp.float32

  # Input Encoder
  for lyr in range(num_layers):
    x = scope.child(encoder_1d_block, f'encoderblock_{lyr}')(
        x,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        dtype=dtype,
        padding_mask=src_padding_mask,
        inputs_segmentation=inputs_segmentation,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        deterministic=not train)
  encoded = scope.child(nn.layer_norm, 'encoder_norm')(x, dtype=dtype)

  return encoded


def decoder(
    scope,
    encoded,
    src_padding_mask,
    targets,
    output_vocab_size,
    targets_positions=None,
    inputs_segmentation=None,
    targets_segmentation=None,
    tgt_padding_mask=None,
    shared_embedding=None,
    logits_via_embedding=False,
    shift=True,
    use_bfloat16=False,
    emb_dim=512,
    num_heads=8,
    num_layers=6,
    qkv_dim=512,
    mlp_dim=2048,
    max_len=512,
    train=True,
    cache=None,
    dropout_rate=0.1,
    attention_dropout_rate=0.1):
  """Applies Transformer model on the inputs.

  Args:
    encoded: encoded input data from encoder.
    src_padding_mask: padding mask for inputs.
    targets: target inputs.
    output_vocab_size: size of the vocabulary.
    targets_positions: input subsequence positions for packed examples.
    inputs_segmentation: input segmentation info for packed examples.
    targets_segmentation: target segmentation info for packed examples.
    tgt_padding_mask: target tokens padding mask.
    shared_embedding: a shared embedding layer to use.
    logits_via_embedding: bool: whether final logit transform shares
      embedding weights.
    shift: whether to shift or not (for fast decoding).
    use_bfloat16: bool: whether use bfloat16.
    emb_dim: dimension of embedding.
    num_heads: number of heads.
    num_layers: number of layers.
    qkv_dim: dimension of the query/key/value.
    mlp_dim: dimension of the mlp on top of attention block.
    max_len: maximum length.
    train: whether it is training.
    cache: flax attention cache for fast decoding.
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate for attention weights.

  Returns:
    output of a transformer decoder.
  """
  assert encoded.ndim == 3  # (batch, len, depth)
  assert targets.ndim == 2  # (batch, len)

  # Padding Masks
  if tgt_padding_mask is None:
    tgt_padding_mask = (targets > 0)[..., None]

  # Target Embedding
  if shared_embedding is None:
    output_embed = scope.child(nn.embedding)(
        num_embeddings=output_vocab_size,
        features=emb_dim,
        init_fn=nn.initializers.normal(stddev=1.0))
  else:
    output_embed = shared_embedding

  y = targets.astype('int32')
  if shift:
    y = shift_right(y)
  y = output_embed.lookup(y)
  y = scope.child(add_position_embs, 'posembed_output')(
      y,
      inputs_positions=targets_positions,
      max_len=max_len,
      cache=cache)
  y = scope.child(nn.dropout)(y, rate=dropout_rate, deterministic=not train)

  if use_bfloat16:
    y = y.astype(jnp.bfloat16)
    dtype = jnp.bfloat16
  else:
    dtype = jnp.float32

  # Target-Input Decoder
  for lyr in range(num_layers):
    y = scope.child(encoder_decoder_1d_block, f'encoderdecoderblock_{lyr}')(
        y,
        encoded,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        dtype=dtype,
        padding_mask=tgt_padding_mask,
        key_padding_mask=src_padding_mask,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        deterministic=not train,
        cache=cache)
  y = scope.child(nn.layer_norm, 'encoderdecoder_norm')(y, dtype=dtype)

  # Decoded Logits
  if logits_via_embedding:
    # Use the transpose of embedding matrix for logit transform.
    logits = output_embed.attend(y.astype(jnp.float32))
    # Correctly normalize pre-softmax logits for this shared case.
    logits = logits / jnp.sqrt(y.shape[-1])
  else:
    logits = scope.child(nn.dense, 'logitdense')(
        y,
        output_vocab_size,
        dtype=dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))
  return logits


def transformer(
    scope,
    inputs,
    targets,
    vocab_size=None,
    output_vocab_size=None,
    inputs_positions=None,
    targets_positions=None,
    inputs_segmentation=None,
    targets_segmentation=None,
    tgt_padding_mask=None,
    share_embeddings=False,
    logits_via_embedding=False,
    use_bfloat16=False,
    emb_dim=512,
    num_heads=8,
    num_layers=6,
    qkv_dim=512,
    mlp_dim=2048,
    max_len=2048,
    train=False,
    shift=True,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
    cache=None):
  """Applies Transformer model on the inputs.

  Args:
    inputs: input data.
    targets: target data.
    vocab_size: size of the input vocabulary.
    output_vocab_size: size of the output vocabulary. If None, the output
      vocabulary size is assumed to be the same as vocab_size.
    inputs_positions: input subsequence positions for packed examples.
    targets_positions: target subsequence positions for packed examples.
    inputs_segmentation: input segmentation info for packed examples.
    targets_segmentation: target segmentation info for packed examples.
    tgt_padding_mask: target tokens padding mask.
    share_embeddings: bool: share embedding layer for inputs and targets.
    logits_via_embedding: bool: whether final logit transform shares
      embedding weights.
    use_bfloat16: bool: whether use bfloat16.
    emb_dim: dimension of embedding.
    num_heads: number of heads.
    num_layers: number of layers.
    qkv_dim: dimension of the query/key/value.
    mlp_dim: dimension of the mlp on top of attention block.
    max_len: maximum length.
    train: whether it is training.
    shift: whether to right-shift targets.
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate for attention weights.
    cache: flax autoregressive cache for fast decoding.

  Returns:
    output of a transformer decoder.
  """
  if output_vocab_size is None:
    output_vocab_size = vocab_size

  if share_embeddings:
    if output_vocab_size is not None:
      assert output_vocab_size == vocab_size, (
          "can't share embedding with different vocab sizes.")
    shared_embedding = scope.child(nn.embedding)(
        num_embeddings=vocab_size,
        features=emb_dim,
        init_fn=nn.initializers.normal(stddev=1.0))
  else:
    shared_embedding = None

  src_padding_mask = (inputs > 0)[..., None]

  encoded = scope.child(encoder, 'encoder')(
      inputs,
      inputs_positions=inputs_positions,
      inputs_segmentation=inputs_segmentation,
      train=train,
      vocab_size=vocab_size,
      shared_embedding=shared_embedding,
      use_bfloat16=use_bfloat16,
      emb_dim=emb_dim,
      num_heads=num_heads,
      num_layers=num_layers,
      qkv_dim=qkv_dim,
      mlp_dim=mlp_dim,
      max_len=max_len,
      dropout_rate=dropout_rate,
      attention_dropout_rate=attention_dropout_rate)

  logits = scope.child(decoder, 'decoder')(
      encoded,
      src_padding_mask,
      targets,
      targets_positions=targets_positions,
      inputs_segmentation=inputs_segmentation,
      targets_segmentation=targets_segmentation,
      tgt_padding_mask=tgt_padding_mask,
      train=train,
      shift=shift,
      cache=cache,
      output_vocab_size=output_vocab_size,
      shared_embedding=shared_embedding,
      logits_via_embedding=logits_via_embedding,
      use_bfloat16=use_bfloat16,
      emb_dim=emb_dim,
      num_heads=num_heads,
      num_layers=num_layers,
      qkv_dim=qkv_dim,
      mlp_dim=mlp_dim,
      max_len=max_len,
      dropout_rate=dropout_rate,
      attention_dropout_rate=attention_dropout_rate)

  return logits.astype(jnp.float32) if use_bfloat16 else logits


def transformer_encode(
    scope,
    inputs,
    vocab_size=None,
    output_vocab_size=None,
    inputs_positions=None,
    inputs_segmentation=None,
    share_embeddings=False,
    logits_via_embedding=True,
    use_bfloat16=False,
    emb_dim=512,
    num_heads=8,
    num_layers=6,
    qkv_dim=512,
    mlp_dim=2048,
    max_len=2048,
    train=False,
    dropout_rate=0.1,
    attention_dropout_rate=0.1):
  """Applies Transformer model on the inputs.

  Args:
    inputs: input data.
    targets: target data.
    vocab_size: size of the input vocabulary.
    output_vocab_size: size of the output vocabulary. If None, the output
      vocabulary size is assumed to be the same as vocab_size.
    inputs_positions: input subsequence positions for packed examples.
    targets_positions: target subsequence positions for packed examples.
    inputs_segmentation: input segmentation info for packed examples.
    targets_segmentation: target segmentation info for packed examples.
    tgt_padding_mask: target tokens padding mask.
    share_embeddings: bool: share embedding layer for inputs and targets.
    logits_via_embedding: bool: whether final logit transform shares
      embedding weights.
    use_bfloat16: bool: whether use bfloat16.
    emb_dim: dimension of embedding.
    num_heads: number of heads.
    num_layers: number of layers.
    qkv_dim: dimension of the query/key/value.
    mlp_dim: dimension of the mlp on top of attention block.
    max_len: maximum length.
    train: whether it is training.
    shift: whether to right-shift targets.
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate for attention weights.
    cache: flax autoregressive cache for fast decoding.

  Returns:
    output of a transformer decoder.
  """
  del logits_via_embedding

  if output_vocab_size is None:
    output_vocab_size = vocab_size

  if share_embeddings:
    if output_vocab_size is not None:
      assert output_vocab_size == vocab_size, (
          "can't share embedding with different vocab sizes.")
    shared_embedding = scope.child(nn.embedding)(
        num_embeddings=vocab_size,
        features=emb_dim,
        init_fn=nn.initializers.normal(stddev=1.0))
  else:
    shared_embedding = None

  encoded = scope.child(encoder, 'encoder')(
      inputs,
      inputs_positions=inputs_positions,
      inputs_segmentation=inputs_segmentation,
      train=train,
      vocab_size=vocab_size,
      shared_embedding=shared_embedding,
      use_bfloat16=use_bfloat16,
      emb_dim=emb_dim,
      num_heads=num_heads,
      num_layers=num_layers,
      qkv_dim=qkv_dim,
      mlp_dim=mlp_dim,
      max_len=max_len,
      dropout_rate=dropout_rate,
      attention_dropout_rate=attention_dropout_rate)

  return encoded


def transformer_decode(
    scope,
    encoded,
    src_padding_mask,
    targets,
    vocab_size=None,
    output_vocab_size=None,
    targets_positions=None,
    inputs_segmentation=None,
    targets_segmentation=None,
    tgt_padding_mask=None,
    share_embeddings=False,
    logits_via_embedding=False,
    use_bfloat16=False,
    emb_dim=512,
    num_heads=8,
    num_layers=6,
    qkv_dim=512,
    mlp_dim=2048,
    max_len=2048,
    train=False,
    shift=True,
    dropout_rate=0.1,
    attention_dropout_rate=0.1,
    cache=None):
  """Applies Transformer model on the inputs.

  Args:
    inputs: input data.
    targets: target data.
    vocab_size: size of the input vocabulary.
    output_vocab_size: size of the output vocabulary. If None, the output
      vocabulary size is assumed to be the same as vocab_size.
    inputs_positions: input subsequence positions for packed examples.
    targets_positions: target subsequence positions for packed examples.
    inputs_segmentation: input segmentation info for packed examples.
    targets_segmentation: target segmentation info for packed examples.
    tgt_padding_mask: target tokens padding mask.
    share_embeddings: bool: share embedding layer for inputs and targets.
    logits_via_embedding: bool: whether final logit transform shares
      embedding weights.
    use_bfloat16: bool: whether use bfloat16.
    emb_dim: dimension of embedding.
    num_heads: number of heads.
    num_layers: number of layers.
    qkv_dim: dimension of the query/key/value.
    mlp_dim: dimension of the mlp on top of attention block.
    max_len: maximum length.
    train: whether it is training.
    shift: whether to right-shift targets.
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate for attention weights.
    cache: flax autoregressive cache for fast decoding.

  Returns:
    output of a transformer decoder.
  """
  if output_vocab_size is None:
    output_vocab_size = vocab_size

  if share_embeddings:
    if output_vocab_size is not None:
      assert output_vocab_size == vocab_size, (
          "can't share embedding with different vocab sizes.")
    shared_embedding = scope.child(nn.embedding)(
        num_embeddings=vocab_size,
        features=emb_dim,
        init_fn=nn.initializers.normal(stddev=1.0))
  else:
    shared_embedding = None

  logits = scope.child(decoder, 'decoder')(
      encoded,
      src_padding_mask,
      targets,
      targets_positions=targets_positions,
      inputs_segmentation=inputs_segmentation,
      targets_segmentation=targets_segmentation,
      tgt_padding_mask=tgt_padding_mask,
      train=train,
      shift=shift,
      cache=cache,
      output_vocab_size=output_vocab_size,
      shared_embedding=shared_embedding,
      logits_via_embedding=logits_via_embedding,
      use_bfloat16=use_bfloat16,
      emb_dim=emb_dim,
      num_heads=num_heads,
      num_layers=num_layers,
      qkv_dim=qkv_dim,
      mlp_dim=mlp_dim,
      max_len=max_len,
      dropout_rate=dropout_rate,
      attention_dropout_rate=attention_dropout_rate)

  return logits.astype(jnp.float32) if use_bfloat16 else logits
