import os
import torch
import pickle
from gym import Space

from CybORG import CybORG
from CybORG.Agents import BaseAgent
from ray.rllib.policy.policy import PolicySpec, Policy

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from CybORG.Agents.Wrappers.EnterpriseMAE import EnterpriseMAE

# Import your custom agents here.
# from __future__ import annotations

class Agent(BaseAgent):
    def __init__(self, name: str = None, model = None):
        super().__init__(name)
        self.policy = model

    def get_action(self, observation: dict, action_space: Space):
        print(">>>shape: ", observation.shape)
        return self.policy.compute_single_action(observation)
    
class Submission:

    # Submission name
    NAME: str = "SUBMISSION NAME"

    # Name of your team
    TEAM: str = "TEAM NAME"

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = "TECHNIQUE NAME"

    # Use this function to define your agents.
    AGENTS: dict[str, BaseAgent] = {
        # f"blue_agent_{agent}": Agent(f'blue_agent_{agent}', pickle.load(open(os.path.dirname(f'/Users/rll249/Documents/CAGE/cage-4-playground/staging/policies/Agent{agent}/') + "/policy_state.pkl", 'rb'))) for agent in range(5)
        # Agent(pickle.load(os.path.dirname(f'/Users/rll249/Documents/CAGE/cage-4-playground/staging/policies/Agent{agent}/') + "/policy_state.pkl")) for agent in range(5)
        # f"blue_agent_{agent}": Agent(torch.load(os.path.dirname(f'/Users/rll249/Documents/CAGE/cage-4-playground/staging/policies/Agent{agent}/') + "/policy_state.pkl")) for agent in range(5)
        f"blue_agent_{agent}": Agent(f'Agent{agent}', Policy.from_checkpoint(os.path.dirname(f'/Users/rll249/Documents/CAGE/cage-4-playground/Submissions/staging/policies/Agent{agent}/'))) for agent in range(5)
    }

    # Use this function to wrap CybORG with your custom wrapper(s).
    def wrap(env: CybORG) -> MultiAgentEnv:
        return EnterpriseMAE(env)