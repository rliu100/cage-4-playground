import os
import torch
import pickle
from gym import Space

from CybORG import CybORG
from CybORG.Agents import BaseAgent
from ray.rllib.policy.policy import PolicySpec, Policy
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.algorithm import Algorithm
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import EnterpriseMAE
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent

from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.policy.policy import PolicySpec

from ray.rllib.env.multi_agent_env import MultiAgentEnv
from CybORG.Agents.Wrappers.EnterpriseMAE import EnterpriseMAE

# Import your custom agents here.
# from __future__ import annotations

class Agent(BaseAgent):
    def __init__(self, name: str = None, model = None):
        super().__init__(name)
        self.model = model

    def get_action(self, observation: dict, action_space: Space):
        return self.model(observation)
    
def env_creator_CC4(env_config: dict):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500
    )
    cyborg = CybORG(scenario_generator=sg)
    env = EnterpriseMAE(env=cyborg)
    return env

NUM_AGENTS = 5
POLICY_MAP = {f"blue_agent_{i}": f"Agent{i}" for i in range(NUM_AGENTS)}

def policy_mapper(agent_id, episode, worker, **kwargs):
# def policy_mapper(agent_id, episode, **kwargs):
    return POLICY_MAP[agent_id]
    
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
        f"blue_agent_{agent}": Agent(Algorithm.from_checkpoint(os.path.dirname(f'/Users/rll249/Documents/CAGE/cage-4-playground/Submissions/no_worker_agent/staging/policies/Agent{agent}/') + "/policy_state.pkl")) for agent in range(5)

    }

    # Use this function to wrap CybORG with your custom wrapper(s).
    def wrap(env: CybORG) -> MultiAgentEnv:
        return EnterpriseMAE(env)
    
    def restore(path):
        register_env(name='CC4', env_creator=lambda config: env_creator_CC4(config))
        env = env_creator_CC4({})
        algo_config = (
            PPOConfig()
            .environment(env="CC4")
            .debugging(logger_config={'logdir':'logs/PPO_Example', 'type': 'ray.tune.logger.TBXLogger'})
            .multi_agent(policies={
                ray_agent: PolicySpec(
                    policy_class=None,
                    observation_space=env.observation_space(cyborg_agent),
                    action_space=env.action_space(cyborg_agent),
                    config={'gamma': 0.85}
                ) for cyborg_agent, ray_agent in POLICY_MAP.items()
            },
            policy_mapping_fn=policy_mapper
        ))

        new_algo = algo_config.build()
        restored_algo = new_algo.restore(os.path.dirname(f'/Users/rll249/Documents/CAGE/cage-4-playground/no_worker_agent/staging/policies/'))
