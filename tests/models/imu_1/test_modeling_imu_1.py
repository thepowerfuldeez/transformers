# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing suite for the PyTorch Imu-1 model."""

import unittest

from transformers import Imu1Config, is_torch_available
from transformers.testing_utils import require_torch, torch_device

from ...test_modeling_common import ModelTesterMixin, _config_zero_init, ids_tensor


if is_torch_available():
    import torch

    from transformers import Imu1ForCausalLM, Imu1Model


class Imu1ModelTester:
    def __init__(
        self,
        parent,
        batch_size=4,
        seq_length=5,
        vocab_size=97,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=48,
        max_position_embeddings=32,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = torch.ones((self.batch_size, self.seq_length), device=torch_device)
        config = self.get_config()
        return config, input_ids, attention_mask

    def get_config(self):
        return Imu1Config(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=self.max_position_embeddings,
            attn_gating="per-head",
            attn_qknorm=True,
            attn_val_residual=True,
        )

    def create_and_check_model(self, config, input_ids, attention_mask):
        model = Imu1Model(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=attention_mask)
        self.parent.assertEqual(result.last_hidden_state.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def create_and_check_lm_head_model(self, config, input_ids, attention_mask):
        model = Imu1ForCausalLM(config=config)
        model.to(torch_device)
        model.eval()
        with torch.no_grad():
            result = model(input_ids=input_ids, attention_mask=attention_mask)
        self.parent.assertEqual(
            result.logits.shape, (self.batch_size, self.seq_length, self.vocab_size)
        )

    def prepare_config_and_inputs_for_common(self):
        config, input_ids, attention_mask = self.prepare_config_and_inputs()
        return config, {"input_ids": input_ids, "attention_mask": attention_mask}


@require_torch
class Imu1ModelTest(ModelTesterMixin, unittest.TestCase):
    all_model_classes = (Imu1Model, Imu1ForCausalLM) if is_torch_available() else ()
    test_pruning = False
    test_head_pruning = False
    test_missing_keys = False
    test_torchscript = False
    test_mismatched_shapes = False
    test_resize_embeddings = True

    def setUp(self):
        self.model_tester = Imu1ModelTester(self)
        self.config_tester = None

    def test_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_model(*config_and_inputs)

    def test_lm_head_model(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        self.model_tester.create_and_check_lm_head_model(*config_and_inputs)

    def test_inputs_embeds(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config, input_ids, attention_mask = config_and_inputs
        inputs_embeds = torch.randn(
            (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.hidden_size),
            device=torch_device,
        )
        model = Imu1Model(config=config).to(torch_device).eval()
        with torch.no_grad():
            result = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        self.assertEqual(
            result.last_hidden_state.shape, (self.model_tester.batch_size, self.model_tester.seq_length, self.model_tester.hidden_size)
        )

    def test_training(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config, input_ids, attention_mask = config_and_inputs
        model = Imu1ForCausalLM(config=config).to(torch_device)
        model.train()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        self.assertIsNotNone(outputs.loss)

    def test_hidden_states_output(self):
        config_and_inputs = self.model_tester.prepare_config_and_inputs()
        config, input_ids, attention_mask = config_and_inputs
        model = Imu1Model(config=config).to(torch_device).eval()
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        self.assertEqual(len(hidden_states), self.model_tester.num_hidden_layers + 1)

    @_config_zero_init
    def test_initialization(self):
        config, input_ids, attention_mask = self.model_tester.prepare_config_and_inputs()
        model = Imu1Model(config=config).to(torch_device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        desired_std = config.initializer_range / (2 * config.num_hidden_layers) ** 0.5
        self.assertLessEqual(abs(hidden_states).mean().item(), desired_std)


if __name__ == "__main__":
    unittest.main()

