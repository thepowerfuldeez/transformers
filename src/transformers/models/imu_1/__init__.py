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
from ...utils import OptionalDependencyNotAvailable, is_torch_available, is_vision_available
from .configuration_imu_1 import Imu1Config


try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    from ...utils.dummy_pt_objects import *  # noqa F403
else:
    from .modeling_imu_1 import *  # noqa F403

__all__ = ["Imu1Config"]
if is_torch_available():
    __all__ += ["Imu1Model", "Imu1ForCausalLM"]

