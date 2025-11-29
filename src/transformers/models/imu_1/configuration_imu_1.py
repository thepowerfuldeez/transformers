# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Imu-1 model configuration."""

from typing import Optional

from ...configuration_utils import PreTrainedConfig
from ...utils import logging


logger = logging.get_logger(__name__)


class Imu1Config(PreTrainedConfig):
    r"""
    This is the configuration class to store the configuration of an [`Imu1Model`]. It is used to instantiate an
    imu-1 model according to the specified arguments, defining the model architecture.

    Configuration objects inherit from [`PreTrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 64000):
            Vocabulary size of the model. Defines the number of different tokens that can be represented by the
            `inputs_ids`.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
        intermediate_size (`int`, *optional*, defaults to 4096):
            Dimension of the MLP representations.
        num_hidden_layers (`int`, *optional*, defaults to 24):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        num_key_value_heads (`int`, *optional*):
            This is the number of key_value heads that should be used to implement Grouped Query Attention. If
            `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if
            `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used.
        hidden_act (`str` or `function`, *optional*, defaults to "silu"):
            The non-linear activation function (function or string) in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 8192):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if `config.is_decoder=True`.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            Base period of the RoPE embeddings.
        rope_scaling (`dict`, *optional*):
            Dictionary containing RoPE scaling parameters. If provided, it follows the same format as other RoPE
            scaling configs in 🤗 Transformers.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        attn_qknorm (`bool`, *optional*, defaults to `True`):
            Whether to apply query/key normalization before attention.
        attn_val_residual (`bool`, *optional*, defaults to `True`):
            Whether to enable value residual mixing (value reuse from the first layer).
        attn_gating (`str` or `bool`, *optional*, defaults to `"per-head"`):
            Attention gating mode. Supported values: `"per-head"`, `"per-head-hd"`, `"elementwise"` or `False`.
        layernorm_scaling (`bool`, *optional*, defaults to `False`):
            Whether to scale RMSNorm outputs by a fixed factor depending on the layer index.
        pad_token_id (`int`, *optional*, defaults to 0):
            Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1):
            Beginning of stream token id.
        eos_token_id (`int`, *optional*, defaults to 2):
            End of stream token id.
    """

    model_type = "imu_1"
    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.qkv_proj": "colwise",
        "layers.*.self_attn.attn_gate": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    @property
    def rope_scaling(self):
        return getattr(self, "_rope_scaling", None)

    @rope_scaling.setter
    def rope_scaling(self, value):
        self._rope_scaling = value

    def __init__(
        self,
        vocab_size: Optional[int] = 64000,
        hidden_size: Optional[int] = 1024,
        intermediate_size: Optional[int] = 4096,
        num_hidden_layers: Optional[int] = 24,
        num_attention_heads: Optional[int] = 16,
        num_key_value_heads: Optional[int] = None,
        hidden_act: Optional[str] = "silu",
        max_position_embeddings: Optional[int] = 8192,
        initializer_range: Optional[float] = 0.02,
        rms_norm_eps: Optional[float] = 1e-5,
        use_cache: Optional[bool] = True,
        rope_theta: Optional[float] = 10000.0,
        rope_scaling: Optional[dict] = None,
        tie_word_embeddings: Optional[bool] = False,
        attn_qknorm: Optional[bool] = True,
        attn_val_residual: Optional[bool] = True,
        attn_gating: Optional[str | bool] = "per-head",
        layernorm_scaling: Optional[bool] = False,
        pad_token_id: Optional[int] = 0,
        bos_token_id: Optional[int] = 1,
        eos_token_id: Optional[int] = 2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_attention_heads if num_key_value_heads is None else num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attn_qknorm = attn_qknorm
        self.attn_val_residual = attn_val_residual
        self.attn_gating = attn_gating
        self.layernorm_scaling = layernorm_scaling
        self.tokenizer_class = "PreTrainedTokenizerFast"

        # Align with standard RoPE handling so helpers like `LlamaRotaryEmbedding`-style utilities work out of the box.
        self.rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}
        if self.rope_scaling is not None:
            self.rope_parameters["rope_scaling"] = self.rope_scaling


__all__ = ["Imu1Config"]
