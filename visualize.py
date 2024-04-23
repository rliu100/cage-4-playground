from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, DiscoveryFSRed, cc4BlueRandomAgent, RandomAgent
from CybORG.Agents.Wrappers.VisualiseRedExpansion import VisualiseRedExpansion
from CybORG.Agents.SimpleAgents.FiniteStateRedAgent import FiniteStateRedAgent

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
from inspect import signature

from CybORG.Evaluation.evaluation import load_submission
from ray.rllib.models import ModelCatalog
from CustomRLLib import TorchActionMaskModel

# Import your custom agents here.
ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)

class Agent(BaseAgent):
    def __init__(self, name: str = None, policy = None):
        super().__init__(name)
        self.policy = policy

    def convert_obs(self, obs):
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
        result = self.policy.compute_single_action(self.convert_obs(observation))
        print("action: ", result[0])
        return result[0]
    
    def end_episode(self):
        pass

    def set_initial_values(self, action_space, observation):
        pass
    
class Submission:

    # Submission name
    NAME: str = "SUBMISSION NAME"

    # Name of your team
    TEAM: str = "TEAM NAME"

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = "TECHNIQUE NAME"

    # Use this function to define your agents.
    AGENTS: dict[str, BaseAgent] = {
        f"blue_agent_{agent}": Agent(f'Agent{agent}', Policy.from_checkpoint(os.path.dirname(f'/Users/rll249/Documents/CAGE/cage-4-playground/Submissions/2_rounds/staging/policies/Agent{agent}/'))) for agent in range(5)
    }

    # Use this function to wrap CybORG with your custom wrapper(s).
    def wrap(env: CybORG) -> MultiAgentEnv:
        return EnterpriseMAE(env)

submission = load_submission('/Users/rll249/Documents/CAGE/cage-4-playground/Submissions/2_rounds/staging')

steps = 200
sg = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent, 
                                green_agent_class=EnterpriseGreenAgent, 
                                red_agent_class=FiniteStateRedAgent,
                                steps=steps)
cyborg = CybORG(scenario_generator=sg, seed=7629)

visualise = VisualiseRedExpansion(cyborg, steps)
visualise.run_evaluation(submission)
# visualise.run()