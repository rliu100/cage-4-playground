
from gym import Space
from CybORG.Agents import BaseAgent

from CybORG import CybORG
from CybORG.Agents import BaseAgent

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from CybORG.Agents.Wrappers.EnterpriseMAE import EnterpriseMAE
from CybORG.Agents import RandomAgent

OBS = []

class DummyAgent(BaseAgent):
    def __init__(self, name: str = None):
        super().__init__(name)

    def get_action(self, observation: dict, action_space: Space):
        OBS.append(observation)
        print(len(OBS))
        return action_space.n - 1


# Import your custom agents here.
# from dummy_agent import DummyAgent
# from __future__ import annotations

class Submission:

    # Submission name
    NAME: str = "SUBMISSION NAME"

    # Name of your team
    TEAM: str = "TEAM NAME"

    # What is the name of the technique used? (e.g. Masked PPO)
    TECHNIQUE: str = "TECHNIQUE NAME"

    # Use this function to define your agents.
    AGENTS: dict[str, BaseAgent] = {
        f"blue_agent_{agent}": DummyAgent() for agent in range(5)
    }

    # Use this function to wrap CybORG with your custom wrapper(s).
    def wrap(env: CybORG) -> MultiAgentEnv:
        return EnterpriseMAE(env)
