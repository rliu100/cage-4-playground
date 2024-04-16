import os
import torch
import pickle
import numpy as np
from gym import Space

from CybORG import CybORG
from CybORG.Agents import BaseAgent
from ray.rllib.policy.policy import PolicySpec, Policy

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from CybORG.Agents.Wrappers.EnterpriseMAE import EnterpriseMAE

from gymnasium.spaces import MultiDiscrete
from ray.rllib.models import ModelCatalog

from TorchActionMaskModel import TorchActionMaskModel
from MaskWrapper import MaskWrapper

# Import your custom agents here.
# from __future__ import annotations

ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)

class Agent(BaseAgent):
    def __init__(self, name: str = None, policy = None):
        super().__init__(name)
        self.policy = policy

    def convert_obs(self, obs):
        """
        For models trained with RLLib using MultiDiscrete Space
        """
        bit_obs_array = []
        if obs[0] == 0:
            bit_obs_array += [1, 0, 0]
        elif obs[0] == 1:
            bit_obs_array += [0, 1, 0]
        else:
            bit_obs_array += [0, 0, 1]

        for i in range(1, len(obs)):
            if obs[i] == 0:
                bit_obs_array += [1, 0]
            else:
                bit_obs_array += [0, 1]
        return np.array(bit_obs_array)

    def get_action(self, observation: dict, action_space: Space):
        obs = {
            'observations': self.convert_obs(observation['observations']),
            'action_mask': observation['action_mask']
        }
        result = self.policy.compute_single_action(obs)
        return result[0]
    
class Submission:

    # Submission name
    NAME: str = "SUBMISSION NAME"

    # Name of your team
    TEAM: str = "TEAM NAME"

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = "TECHNIQUE NAME"

    # Use this function to define your agents.
    AGENTS: dict[str, BaseAgent] = {
        f"blue_agent_{agent}": Agent(f'Agent{agent}', Policy.from_checkpoint(os.path.dirname(f'/Users/rll249/Documents/CAGE/cage-4-playground/Submissions/mask_results_2024-04-16_02:01:29/staging/policies/Agent{agent}/'))) for agent in range(5)
    }

    # Use this function to wrap CybORG with your custom wrapper(s).
    def wrap(env: CybORG) -> MultiAgentEnv:
        return MaskWrapper(env)