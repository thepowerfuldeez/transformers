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
"""PyTorch imu_1 model."""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn

from ...cache_utils import Cache, DynamicCache
from ...generation import GenerationMixin
from ...masking_utils import create_causal_mask
from ...modeling_layers import GradientCheckpointingLayer
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from ...processing_utils import Unpack
from ...utils import (
    ModelOutput,
    TransformersKwargs,
    auto_docstring,
    can_return_tuple,
    logging,
)
from ...utils.generic import check_model_inputs
from ..llama.modeling_llama import eager_attention_forward, repeat_kv
from .configuration_imu_1 import Imu1Config


logger = logging.get_logger(__name__)


def _init_weights(module: nn.Module):
    """Weight init helper used by post_init."""
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=module.weight.shape[-1] ** -0.5)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Imu1RMSNorm(nn.Module):
    """RMSNorm with an optional fixed position scaling factor."""

    def __init__(self, hidden_size: int, eps: float = 1e-5, position_scale: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.register_buffer("position_scale", torch.tensor(position_scale, dtype=torch.float32), persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states
        return hidden_states.to(input_dtype) * self.position_scale.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, scale={float(self.position_scale)}"


class Imu1QKNorm(nn.Module):
    """Lightweight query/key normalization with a learned gain."""

    def __init__(self, head_dim: int):
        super().__init__()
        # Initialize gain to 1.0 to match reference implementation
        self.gain = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q_dtype, k_dtype = q.dtype, k.dtype
        qf, kf = q.float(), k.float()
        # Use eps=1e-6 to match reference implementation
        qf = qf * torch.rsqrt(qf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        kf = kf * torch.rsqrt(kf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        gain = self.gain.to(qf)
        return (qf * gain).to(q_dtype), kf.to(k_dtype)


class Imu1RotaryEmbedding(nn.Module):
    """GPT-J style RoPE with cached cos/sin tables."""

    def __init__(self, config: Imu1Config):
        super().__init__()
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.base = config.rope_theta
        self.max_seq_len_cached = config.max_position_embeddings
        cos, sin = self._build_cache(self.max_seq_len_cached)
        # NOTE: persistent=True to work around transformers bug where non-persistent buffers
        # get replaced with empty tensors during from_pretrained()
        self.register_buffer("cos_cached", cos, persistent=True)
        self.register_buffer("sin_cached", sin, persistent=True)

    def _build_cache(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.cos_cached.device if hasattr(self, "cos_cached") else torch.device("cpu")
        dtype = self.cos_cached.dtype if hasattr(self, "cos_cached") else torch.float32
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        dim = torch.arange(0, self.head_dim // 2, device=device, dtype=torch.float32)
        freq = torch.outer(positions, self.base ** (-2 * dim / self.head_dim))
        return freq.cos().to(dtype), freq.sin().to(dtype)

    def _maybe_extend_cache(self, seq_len: int) -> None:
        if seq_len <= self.max_seq_len_cached:
            return
        cos, sin = self._build_cache(seq_len)
        self.max_seq_len_cached = seq_len
        self.register_buffer("cos_cached", cos, persistent=True)
        self.register_buffer("sin_cached", sin, persistent=True)

    def forward(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = hidden_states.shape[:2]
        if position_ids is None:
            max_pos = seq_len
        else:
            max_pos = int(position_ids.max()) + 1
        self._maybe_extend_cache(max_pos)

        if position_ids is None:
            cos = self.cos_cached[:seq_len].unsqueeze(0)
            sin = self.sin_cached[:seq_len].unsqueeze(0)
        else:
            cos = self.cos_cached[position_ids]
            sin = self.sin_cached[position_ids]

        cos = cos.to(device=hidden_states.device, dtype=hidden_states.dtype)
        sin = sin.to(device=hidden_states.device, dtype=hidden_states.dtype)
        return cos, sin


def apply_imu1_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    def _rotate(x: torch.Tensor) -> torch.Tensor:
        # Use split-half format (LLaMA style, rope_interleaved=False)
        # Split the last dimension in half and rotate (x[:half], x[half:]) pairs
        x_f = x.float()
        half = x_f.size(-1) // 2
        x1, x2 = x_f[..., :half], x_f[..., half:]
        cos_view = cos.unsqueeze(1)  # (bsz, 1, seq_len, head_dim//2)
        sin_view = sin.unsqueeze(1)
        row1 = x1 * cos_view - x2 * sin_view
        row2 = x1 * sin_view + x2 * cos_view
        return torch.cat([row1, row2], dim=-1).to(dtype=x.dtype)

    return _rotate(q), _rotate(k)


class Imu1Attention(nn.Module):
    def __init__(self, config: Imu1Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = 0.0
        self.is_causal = True

        self.qkv_proj = nn.Linear(config.hidden_size, self.q_size + 2 * self.kv_size, bias=False)
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)

        gating_cfg = config.attn_gating
        if gating_cfg and gating_cfg is True:
            gating_cfg = "per-head"
        self.gating = gating_cfg
        self.attn_gate = None
        if self.gating:
            if self.gating == "elementwise":
                gate_out = config.hidden_size
            elif self.gating in ("per-head", "per-head-hd"):
                gate_out = self.num_heads
            else:
                raise ValueError(f"Unsupported gating mode: {self.gating}")
            self.attn_gate = nn.Linear(config.hidden_size if self.gating != "per-head-hd" else self.head_dim, gate_out, bias=False)

        # qk-norm is parameterized by head dimension (see reference implementation)
        self.qk_norm = Imu1QKNorm(self.head_dim) if config.attn_qknorm else None
        if config.attn_val_residual:
            # Initialize as parameters that can be trained
            # Start with alpha1=1.0, alpha2=0.0 to match reference (not 0.5, 0.5)
            self.alpha1 = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
            self.alpha2 = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
            self.value_scale = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        else:
            self.register_buffer("alpha1", torch.tensor([1.0]), persistent=False)
            self.register_buffer("alpha2", torch.tensor([0.0]), persistent=False)
            self.register_buffer("value_scale", torch.tensor([1.0]), persistent=False)
        self.value_norm_eps = 1e-8

    def _mix_values(self, v: torch.Tensor, value_residual: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        v1 = v if value_residual is None else value_residual.view_as(v)
        denom = torch.rsqrt(self.alpha1.square() + self.alpha2.square() + self.value_norm_eps).to(v)
        mixed = self.value_scale.to(v) * (self.alpha1.to(v) * v + self.alpha2.to(v) * v1) * denom
        return mixed, mixed

    def _apply_gating(self, attn_output: torch.Tensor, gate_input: torch.Tensor) -> torch.Tensor:
        if self.attn_gate is None:
            return attn_output

        if self.gating == "per-head-hd":
            gate_input = gate_input[..., : self.head_dim]
        gate = self.attn_gate(gate_input).sigmoid() * 2.0
        if self.gating == "elementwise":
            return attn_output * gate

        gate = gate.view(*attn_output.shape[:2], -1, 1)  # (b, seq, h, 1)
        attn_output = attn_output.view(attn_output.shape[0], attn_output.shape[1], self.num_heads, self.head_dim)
        attn_output = gate * attn_output
        return attn_output.reshape(attn_output.shape[0], attn_output.shape[1], -1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        value_residual: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        q, k = apply_imu1_rotary_pos_emb(q, k, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)

        # Repeat KV heads BEFORE QKNorm and value mixing (matches native implementation)
        key_states = repeat_kv(k, self.num_key_value_groups)
        v_repeated = repeat_kv(v, self.num_key_value_groups)

        mixed_v, value_residual_out = self._mix_values(v_repeated, value_residual)

        # Apply QKNorm after repeating (matches native implementation)
        if self.qk_norm is not None:
            q, key_states = self.qk_norm(q, key_states)

        value_states = mixed_v

        # IMPORTANT: Native implementation uses (heads*batch, seq, head_dim) layout for SDPA
        # We must match this layout to get identical results (heads first, not batch first!)
        # Current shapes: q (bsz, num_heads, q_len, head_dim), key_states/value_states (bsz, num_heads, k_len, head_dim)
        # Target: (num_heads*bsz, seq_len, head_dim) - NOTE: heads*bsz, not bsz*heads!
        k_len = key_states.size(2)  # Key/value sequence length may differ from q_len during generation
        q = q.permute(1, 0, 2, 3).reshape(self.num_heads * bsz, q_len, self.head_dim)
        key_states = key_states.permute(1, 0, 2, 3).reshape(self.num_heads * bsz, k_len, self.head_dim)
        value_states = value_states.permute(1, 0, 2, 3).reshape(self.num_heads * bsz, k_len, self.head_dim)

        # Use SDPA with native layout
        if attention_mask is not None:
            # Need to reshape attention mask to match
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, key_states, value_states,
                attn_mask=attention_mask,
                dropout_p=0.0,
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, key_states, value_states,
                dropout_p=0.0,
                is_causal=True,
            )

        attn_weights = None  # SDPA doesn't return weights
        # Reshape back from (num_heads*bsz, q_len, head_dim) to (bsz, q_len, num_heads*head_dim)
        attn_output = attn_output.reshape(self.num_heads, bsz, q_len, self.head_dim).permute(1, 2, 0, 3).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.q_size)
        attn_output = self._apply_gating(attn_output, hidden_states)
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            attn_weights = attn_weights
        else:
            attn_weights = None

        return attn_output, attn_weights, value_residual_out


class Imu1MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        if hidden_act != "silu":
            raise ValueError(f"Unsupported activation {hidden_act}. Only silu is available for imu_1.")
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(hidden_states)
        x1, x2 = x.chunk(2, dim=-1)
        x = self.act_fn(x1) * x2
        return self.down_proj(x)


class Imu1DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Imu1Config, layer_idx: int):
        super().__init__()
        position_scale = (
            (float(layer_idx + 1)) ** -0.5 if getattr(config, "layernorm_scaling", False) else 1.0
        )
        self.self_attn = Imu1Attention(config, layer_idx)
        self.mlp = Imu1MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = Imu1RMSNorm(config.hidden_size, eps=config.rms_norm_eps, position_scale=position_scale)
        self.post_attention_layernorm = Imu1RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, position_scale=position_scale
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        past_key_values: Optional[Cache],
        cache_position: Optional[torch.Tensor],
        output_attentions: bool,
        value_residual: torch.Tensor | None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_output, attn_weights, value_residual_out = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
            cache_position=cache_position,
            output_attentions=output_attentions,
            value_residual=value_residual,
            **kwargs,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attn_weights, value_residual_out


@auto_docstring
class Imu1PreTrainedModel(PreTrainedModel):
    config_class = Imu1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Imu1DecoderLayer"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_record_outputs = {
        "hidden_states": Imu1DecoderLayer,
        "attentions": Imu1Attention,
    }

    def _init_weights(self, module: nn.Module):
        _init_weights(module)


@auto_docstring
class Imu1Model(Imu1PreTrainedModel):
    def __init__(self, config: Imu1Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([Imu1DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = Imu1RMSNorm(config.hidden_size, eps=config.rms_norm_eps, position_scale=1.0)
        self.rotary_emb = Imu1RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.post_init()

    @check_model_inputs()
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            batch_size, seq_length, _ = inputs_embeds.shape
            if input_ids is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        elif past_key_values is not None and not isinstance(past_key_values, Cache):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = None
        if attention_mask is not None:
            causal_mask = create_causal_mask(
                config=self.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        value_residual: torch.Tensor | None = None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values if use_cache else None,
                cache_position=cache_position,
                output_attentions=output_attentions,
                value_residual=value_residual,
                **kwargs,
            )
            hidden_states, attn_weights, layer_value_residual = layer_outputs
            if value_residual is None:
                value_residual = layer_value_residual

            if output_attentions:
                all_self_attentions += (attn_weights,)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if not return_dict:
            outputs = (hidden_states, next_cache)
            if output_hidden_states:
                outputs += (all_hidden_states,)
            if output_attentions:
                outputs += (all_self_attentions,)
            return outputs

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@auto_docstring
class Imu1ForCausalLM(Imu1PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config: Imu1Config):
        super().__init__(config)
        self.model = Imu1Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        outputs: ModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits, outputs.past_key_values)
            if output_hidden_states:
                output += (outputs.hidden_states,)
            if output_attentions:
                output += (outputs.attentions,)
            if loss is not None:
                output = (loss,) + output
            return output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            if attention_mask is not None and attention_mask[..., -1].sum() == attention_mask.shape[0]:
                input_ids = input_ids[:, -1:]
                if inputs_embeds is not None:
                    inputs_embeds = inputs_embeds[:, -1:]
            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length()
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + input_ids.shape[1], device=input_ids.device
                )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "cache_position": cache_position,
        }


__all__ = ["Imu1Config", "Imu1Model", "Imu1ForCausalLM"]
