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

"""Transformer-based machine translation model."""

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Any

from flax import nn
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


@dataclass
class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs."""
  max_len: int = 512
  posemb_init: Callable = None
  cache: Any = None
  name: Optional[str] = None

  def apply(self, inputs, inputs_positions=None):
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
    pos_emb_shape = (1, self.max_len, inputs.shape[-1])
    if self.posemb_init is None:
      # Use a fixed (non-learned) sinusoidal position embedding.
      pos_embedding = sinusoidal_init(
          max_len=self.max_len)(None, pos_emb_shape, None)
    else:
      pos_embedding = self.param('pos_embedding', pos_emb_shape, self.posemb_init)
    pe = pos_embedding[:, :length, :]
    # We abuse the same attention Cache mechanism to run positional embeddings
    # in fast predict mode. We could use state variables instead, but this
    # simplifies invocation with a single top-level cache context manager.
    # We only use the cache's position index for tracking decoding position.
    if self.cache:
      if self.is_initializing():
        self.cache.store(lambda: (4, (1, 1)))
      else:
        cache_entry = self.cache.retrieve(None)
        i = cache_entry.i
        self.cache.store(cache_entry.replace(i=cache_entry.i + 1))
        _, _, df = pos_embedding.shape
        pe = lax.dynamic_slice(pos_embedding,
                               jnp.array((0, i, 0)),
                               jnp.array((1, 1, df)))
    if inputs_positions is None:
      # normal unpacked case:
      return inputs + pe
    else:
      # for packed data we need to use known position indices:
      return inputs + jnp.take(pe[0], inputs_positions, axis=0)


@dataclass
class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  mlp_dim: int
  dtype: Any = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  deterministic: bool = False
  kernel_init: Callable = nn.initializers.xavier_uniform()
  bias_init: Callable = nn.initializers.normal(stddev=1e-6)
  name: Optional[str] = None

  def apply(self, inputs):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(self.mlp_dim, dtype=self.dtype, kernel_init=self.kernel_init,
                 bias_init=self.bias_init)(inputs)
    x = nn.relu(x)
    x = nn.dropout(x, rate=self.dropout_rate, deterministic=self.deterministic)
    output = nn.Dense(actual_out_dim, dtype=self.dtype, kernel_init=self.kernel_init,
                      bias_init=self.bias_init)(x)
    output = nn.dropout(output, rate=self.dropout_rate, deterministic=self.deterministic)
    return output


@dataclass
class Encoder1DBlock(nn.Module):
  """Transformer decoder layer."""
  qkv_dim: int
  mlp_dim: int
  num_heads: int
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  name: Optional[str] = None

  def apply(self,
            inputs,
            inputs_segmentation=None,
            padding_mask=None):
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
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_dim,
        attention_axis=(1,),
        causal_mask=False,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        deterministic=self.deterministic)(
            inputs_q=x,
            inputs_kv=x,
            padding_mask=padding_mask,
            segmentation=inputs_segmentation)
    x = nn.dropout(x, rate=self.dropout_rate, deterministic=self.deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        deterministic=self.deterministic)(y)

    return x + y


@dataclass
class EncoderDecoder1DBlock(nn.Module):
  """Transformer encoder-decoder layer."""
  qkv_dim: int
  mlp_dim: int
  num_heads: int
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  deterministic: bool = False
  cache: Any = None
  name: Optional[str] = None

  def apply(self,
            targets,
            encoded,
            inputs_segmentation=None,
            targets_segmentation=None,
            padding_mask=None,
            key_padding_mask=None):
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
    x = nn.LayerNorm(dtype=self.dtype)(targets)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_dim,
        attention_axis=(1,),
        causal_mask=True,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        deterministic=self.deterministic,
        cache=self.cache)(
            inputs_q=x,
            inputs_kv=x,
            padding_mask=padding_mask,
            segmentation=targets_segmentation)
    x = nn.dropout(x, rate=self.dropout_rate, deterministic=self.deterministic)
    x = x + targets

    # Encoder-Decoder block.
    y = nn.LayerNorm(type=self.dtype)(x)
    y = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        qkv_features=self.qkv_dim,
        attention_axis=(1,),
        causal_mask=False,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
        bias=False,
        broadcast_dropout=False,
        dropout_rate=self.attention_dropout_rate,
        deterministic=self.deterministic)(
            inputs_q=y,
            inputs_kv=encoded,
            padding_mask=padding_mask,
            key_padding_mask=key_padding_mask,
            segmentation=targets_segmentation,
            key_segmentation=inputs_segmentation,
        )
    y = nn.dropout(y, rate=self.dropout_rate, deterministic=self.deterministic)
    y = y + x

    # MLP block.
    z = nn.LayerNorm(dtype=self.dtype)(y)
    z = MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        deterministic=self.deterministic)(z)

    return y + z


@dataclass
class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  vocab_size: int
  shared_embedding: bool = None
  use_bfloat16: bool = False
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 512
  train: bool = True
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  name: Optional[str] = None

  def apply(self,
            inputs,
            inputs_positions=None,
            inputs_segmentation=None):
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
    if self.shared_embedding is None:
      input_embed = nn.Embed(
          num_embeddings=self.vocab_size,
          features=self.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      input_embed = self.shared_embedding
    x = inputs.astype('int32')
    x = input_embed(x)
    x = AddPositionEmbs(
        inputs_positions=inputs_positions,
        max_len=self.max_len,
        name='posembed_input')(x)
    x = nn.dropout(x, rate=self.dropout_rate, deterministic=not self.train)

    if self.use_bfloat16:
      x = x.astype(jnp.bfloat16)
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Input Encoder
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          x,
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dtype=dtype,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          deterministic=not self.train,
          name=f'encoderblock_{lyr}')(
              x,
              padding_mask=src_padding_mask,
              inputs_segmentation=inputs_segmentation)
    encoded = nn.LayerNorm(dtype=dtype, name='encoder_norm')(x)

    return encoded


@dataclass
class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation."""
  output_vocab_size: int
  shared_embedding: bool = None
  use_bfloat16: bool = False
  logits_via_embedding: bool = False
  shift: bool = True
  emb_dim: int = 512
  num_heads: int = 8
  num_layers: int = 6
  qkv_dim: int = 512
  mlp_dim: int = 2048
  max_len: int = 512
  train: bool = True
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  cache: Any = None
  name: Optional[str] = None

  def apply(self,
            encoded,
            src_padding_mask,
            targets,
            targets_positions=None,
            inputs_segmentation=None,
            targets_segmentation=None,
            tgt_padding_mask=None):
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
    if self.shared_embedding is None:
      output_embed = nn.Embed(
          num_embeddings=self.output_vocab_size,
          features=self.emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      output_embed = self.shared_embedding

    y = targets.astype('int32')
    if self.shift:
      y = shift_right(y)
    y = output_embed(y)
    y = AddPositionEmbs(
        inputs_positions=targets_positions,
        max_len=self.max_len,
        cache=self.cache,
        name='posembed_output')(y)
    y = nn.dropout(y, rate=self.dropout_rate, deterministic=not self.train)

    if self.use_bfloat16:
      y = y.astype(jnp.bfloat16)
      dtype = jnp.bfloat16
    else:
      dtype = jnp.float32

    # Target-Input Decoder
    for lyr in range(self.num_layers):
      y = EncoderDecoder1DBlock(
          qkv_dim=self.qkv_dim,
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dtype=dtype,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          deterministic=not self.train,
          cache=self.cache,
          name=f'encoderdecoderblock_{lyr}')(
              y,
              encoded,
              padding_mask=tgt_padding_mask,
              key_padding_mask=src_padding_mask,
              inputs_segmentation=inputs_segmentation,
              targets_segmentation=targets_segmentation,
          )
    y = nn.LayerNorm(dtype=dtype, name='encoderdecoder_norm')(y)

    # Decoded Logits
    if self.logits_via_embedding:
      # Use the transpose of embedding matrix for logit transform.
      logits = output_embed.attend(y.astype(jnp.float32))
      # Correctly normalize pre-softmax logits for this shared case.
      logits = logits / jnp.sqrt(y.shape[-1])
    else:
      logits = nn.Dense(
          self.output_vocab_size,
          dtype=dtype,
          kernel_init=nn.initializers.xavier_uniform(),
          bias_init=nn.initializers.normal(stddev=1e-6),
          name='logitdense')(y)
    return logits


class Transformer(nn.Module):
  """Transformer Model for sequence to sequence translation."""
  def __init__(
      self,
      vocab_size=None,
      output_vocab_size=None,
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
      cache=None,
      name=None):
    self.name = name
    self.use_bfloat16 = use_bfloat16

    if share_embeddings:
      if output_vocab_size is not None:
        assert output_vocab_size == vocab_size, (
            "can't share embedding with different vocab sizes.")
      self.shared_embedding = nn.Embed(
          num_embeddings=vocab_size,
          features=emb_dim,
          embedding_init=nn.initializers.normal(stddev=1.0))
    else:
      self.shared_embedding = None

    self.encoder = Encoder(
        vocab_size=vocab_size,
        shared_embedding=self.shared_embedding,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        name='encoder')

    self.decoder = Decoder(
        output_vocab_size=output_vocab_size,
        shared_embedding=self.shared_embedding,
        logits_via_embedding=logits_via_embedding,
        use_bfloat16=use_bfloat16,
        emb_dim=emb_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        qkv_dim=qkv_dim,
        mlp_dim=mlp_dim,
        max_len=max_len,
        train=train,
        shift=shift,
        dropout_rate=dropout_rate,
        attention_dropout_rate=attention_dropout_rate,
        cache=cache,
        name='decoder')


  def apply(self,
            inputs,
            targets,
            inputs_positions=None,
            targets_positions=None,
            inputs_segmentation=None,
            targets_segmentation=None,
            tgt_padding_mask=None):
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
    src_padding_mask = (inputs > 0)[..., None]

    encoded = self.encoder(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation)

    logits = self.decoder(
        encoded,
        src_padding_mask,
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        tgt_padding_mask=tgt_padding_mask)

    return logits.astype(jnp.float32) if self.use_bfloat16 else logits

  def encode(self,
             inputs,
             inputs_positions=None,
             inputs_segmentation=None):

    encoded = self.encoder(
        inputs,
        inputs_positions=inputs_positions,
        inputs_segmentation=inputs_segmentation)

    return encoded

  def decode(self,
             encoded,
             src_padding_mask,
             targets,
             targets_positions=None,
             inputs_segmentation=None,
             targets_segmentation=None,
             tgt_padding_mask=None):

    logits = self.decoder(
        encoded,
        src_padding_mask,
        targets,
        targets_positions=targets_positions,
        inputs_segmentation=inputs_segmentation,
        targets_segmentation=targets_segmentation,
        tgt_padding_mask=tgt_padding_mask)

    return logits
