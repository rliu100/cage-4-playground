import torch
import torch.nn as nn
import numpy as np
import ray
from ray.rllib.algorithms import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC

from gymnasium.spaces import Dict, Box

from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MIN

tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

class TorchActionMaskModel(TorchModelV2, nn.Module):
    """
    PyTorch version of above ActionMaskingModel.
    from: https://github.com/ray-project/ray/blob/master/rllib/examples/_old_api_stack/models/action_mask_model.py
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        # raise ValueError(f">>== obs_spaces {vars(obs_space)}")
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, Dict)
            and "action_mask" in orig_space.spaces
            and "observations" in orig_space.spaces
        )
        # orig_space = obs_space['obs']
        # raise ValueError(f">>== orig_space.spaces {orig_space['observations']}")

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kwargs
        )
        nn.Module.__init__(self)

        # self.input_size = obs_space.shape[0]
        # self.action_space_size = action_space.n
        # raise ValueError(f'int(np.product(obs_space.shape)):  {int(np.product(orig_space["observations"].shape))}, \n orig_space["observations"]: {orig_space["observations"]}')

        observations_shape = orig_space["observations"].shape[0]
        model_obs_shape = 3+(observations_shape-1)*2
        self.internal_model = TorchFC(
            Box(-1.0, 1.0, (model_obs_shape,), float),
            # obs_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        self.no_masking = False
        if "no_masking" in model_config["custom_model_config"]:
            self.no_masking = model_config["custom_model_config"]["no_masking"]

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        # try:
        # print("!!input_dict obs shape: ", input_dict['obs']['observations'].shape)
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        if self.no_masking:
            return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state
        # except:
        #     raise ValueError(f"input_dict[obs][observations]:  {input_dict['obs']['observations'].shape}, \ninput_dict[obs][action_mask]: {input_dict['obs']['action_mask'].shape}")

    def value_function(self):
        return self.internal_model.value_function()
    
# class ActionMaskModel(TorchModelV2, nn.Module):
#     """
#     PyTorch version of above ActionMaskingModel.
#     from: https://github.com/ray-project/ray/blob/master/rllib/examples/_old_api_stack/models/action_mask_model.py
#     """

#     def __init__(
#         self,
#         obs_space,
#         action_space,
#         num_outputs,
#         model_config,
#         name,
#         **kwargs,
#     ):
#         # raise ValueError(f">>== obs_spaces {vars(obs_space)}")
#         # orig_space = getattr(obs_space, "original_space", obs_space)
#         # assert (
#         #     isinstance(orig_space, Dict)
#         #     and "action_mask" in orig_space.spaces
#         #     and "observations" in orig_space.spaces
#         # )
#         # orig_space = obs_space['obs']
#         # raise ValueError(f">>== orig_space.spaces {orig_space['observations']}")

#         TorchModelV2.__init__(
#             self, obs_space, action_space, num_outputs, model_config, name, **kwargs
#         )
#         nn.Module.__init__(self)

#         # self.input_size = obs_space.shape[0]
#         # self.action_space_size = action_space.n
#         # raise ValueError(f'int(np.product(obs_space.shape)):  {int(np.product(orig_space["observations"].shape))}, \n orig_space["observations"]: {orig_space["observations"]}')

#         # observations_shape = orig_space["observations"].shape[0]
#         # model_obs_shape = 3+(observations_shape-1)*2
#         self.internal_model = TorchFC(
#             # Box(-1.0, 1.0, (model_obs_shape,), float),
#             obs_space,
#             action_space,
#             num_outputs,
#             model_config,
#             name + "_internal",
#         )

#         # disable action masking --> will likely lead to invalid actions
#         self.no_masking = False
#         if "no_masking" in model_config["custom_model_config"]:
#             self.no_masking = model_config["custom_model_config"]["no_masking"]

#     def forward(self, input_dict, state, seq_lens):
#         # Extract the available actions tensor from the observation.
#         # try:
#         action_mask = input_dict["infos"]["action_mask"]

#         # Compute the unmasked logits.
#         logits, _ = self.internal_model({"obs": input_dict["obs"]})

#         # If action masking is disabled, directly return unmasked logits
#         if self.no_masking:
#             return logits, state

#         # Convert action_mask into a [0.0 || -inf]-type mask.
#         inf_mask = torch.clamp(torch.log(action_mask), min=FLOAT_MIN)
#         masked_logits = logits + inf_mask

#         # Return masked logits.
#         return masked_logits, state
#         # except:
#         #     raise ValueError(f"input_dict[obs][observations]:  {input_dict['obs']['observations'].shape}, \ninput_dict[obs][action_mask]: {input_dict['obs']['action_mask'].shape}")

#     def value_function(self):
#         return self.internal_model.value_function()
    

