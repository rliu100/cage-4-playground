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

class CustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name, *args, **kwargs)
        nn.Module.__init__(self)

        # self.fc = TorchFC(obs_space, action_space, num_outputs,
                                    #    model_config, name)
        # raise ValueError("obs_space: ", action_space)
        input_size = obs_space.shape[0]
        action_space_size = action_space.n
        self.input_layer = nn.Linear(input_size, 256)
        self.hidden_layer = nn.Linear(256, 256)
        self.output_dims_layer = nn.Linear(256, action_space_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_dict, state, seq_lens):
        observation = input_dict['obs']
        # rew = input_dict['rew']
        # terminated = input_dict['rew']
        # truncated = input_dict['truncated']
        # info = input_dict['info']
        # mask = input_dict['mask']
        try:
            obs = observation[:, :185]
            mask = observation[:, 185:]
            batch_size, length = mask.shape
            manual_mask_count = 0
            if length == 0:
                manual_mask_count += 1
                mask = torch.ones((batch_size, 82))
          
            x = self.input_layer(obs)
            x = self.hidden_layer(x)
            x = self.output_dims_layer(x)
            x = torch.where(mask == 0, -float('inf'), x)
            x = self.softmax(x)
        except:
            raise ValueError(f"observation shape: {observation.shape}, \nobs type: {type(obs)}, \nobs shape; {obs.shape}, \nmask shape: {mask.shape}, \nmanual_mask: {manual_mask_count}")
        return x, []

    def value_function(self):
        pass