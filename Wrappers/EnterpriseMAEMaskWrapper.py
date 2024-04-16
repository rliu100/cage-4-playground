from __future__ import annotations
from typing import Any

import numpy as np

from CybORG.Agents.Wrappers import BlueEnterpriseWrapper, BlueFixedActionWrapper
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from CybORG import CybORG

import functools
import itertools

from CybORG.Agents.Wrappers.BlueFixedActionWrapper import (
    BlueFixedActionWrapper,
    MESSAGE_LENGTH,
    EMPTY_MESSAGE,
    NUM_MESSAGES,
)

from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import EnterpriseScenarioGenerator

NUM_SUBNETS = 9
NUM_HQ_SUBNETS = 3

MAX_USER_HOSTS = EnterpriseScenarioGenerator.MAX_USER_HOSTS
MAX_SERVER_HOSTS = EnterpriseScenarioGenerator.MAX_SERVER_HOSTS
MAX_HOSTS = MAX_USER_HOSTS + MAX_SERVER_HOSTS

from gymnasium import Space, spaces

class EnterpriseMAEMaskWrapper(BlueEnterpriseWrapper, MultiAgentEnv):
    """A wrapper designed to support CAGE Challenge 4 (RLlib Compatible).

    Creates a vector output for a neural network by directly pulling
    information out of the state object.
    """

    def _get_init_obs_spaces(self):
        """Calculates the size of the largest observation space for each agent."""
        observation_space_components = {
            "mission": [3],
            "blocked_subnets": NUM_SUBNETS * [2],
            "comms_policy": NUM_SUBNETS * [2],
            "malicious_processes": MAX_HOSTS * [2],
            "network_connections": MAX_HOSTS * [2],
            "subnet": NUM_SUBNETS * [2],
            "messages": (NUM_MESSAGES * MESSAGE_LENGTH) * [2],
        }

        observation_head = observation_space_components["mission"]
        observation_tail = observation_space_components["messages"]
        observation_middle = list(
            itertools.chain(
                *[
                    v
                    for k, v in observation_space_components.items()
                    if k not in ("mission", "messages")
                ]
            )
        )

        short_observation_components = (
            observation_head + observation_middle + observation_tail
        )

        long_observation_components = (
            observation_head + (NUM_HQ_SUBNETS * observation_middle) + observation_tail
        )

        short_observation_space = spaces.MultiDiscrete(short_observation_components)
        long_observation_space = spaces.MultiDiscrete(long_observation_components)

        self._observation_space = spaces.Dict({
            agent: spaces.Dict({"observations": long_observation_space, "action_mask": spaces.MultiBinary(self.action_space(agent).n)})
            if self.is_padded or agent == "blue_agent_4"
            else spaces.Dict({"observations": short_observation_space, "action_mask": spaces.MultiBinary(self.action_space(agent).n)})
            for agent in self.agents
        })

        return short_observation_space, long_observation_space

    def step(
        self,
        action_dict: dict[str, Any] | None = None,
        messages: dict[str, Any] | None = None,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Take a step in the enviroment using action indices.

        This wrapper supports both the CybORG and original EnterpriseMAE
        parameter conventions. For example:

            actions = { "blue_agent_0": 42 }
            messages = { "blue_agent_0": np.array([1, 0, 0, 0, 0, 0, 0, 0] }

            # CybORG Convention (preferred)
            env.step(action_dict=actions, messages=messages)

            # EnterpriseMAE Convention
            env.step({
                "actions": actions,
                "messages": messages,
            })

        Args:

            action_dict (dict[str, int]): The action index corresponding to each
                agent. These indices will be mapped to CybORG actions using the
                equivalent of `actions(agent)[index]`. The meaning of each action
                can be found using `action_labels(agent)[index]`.

            messages (dict[str, Any]): Optional messages to be passed to each agent.

            **kwargs (dict[str, Any]): Extra keywords are forwarded.

        Returns:
            observation (dict[str, np.ndarray]): Observations for each agent as vectors.

            rewards (dict[str, float]): Rewards for each agent.

            terminated (dict[str, bool]): Flags whether the agent finished normally.

            truncated (dict[str, bool]): Flags whether the agent was stopped by env.

            info (dict[str, dict]): Forwarded from BlueFixedActionWrapper.
        """
        obs, rew, terminated, truncated, info = super(BlueEnterpriseWrapper, self).step(
            actions=action_dict, messages=messages
        )
        terminated["__all__"] = False
        truncated["__all__"] = self.env.environment_controller.determine_done()

        obs = ({a: {
            "observations": obs[a], "action_mask": 1*np.array(self.action_mask(a))
        } for a in self.agents})

        assert isinstance(obs, dict)
        if (len(info.keys()) != len(obs.keys())):
            raise ValueError(f"len info: {len(info.keys())}, len obs: {len(obs.keys())}, terminated: {terminated}")
        assert len(info.keys()) == len(obs.keys())
        return obs, rew, terminated, truncated, info #, mask
    
    def reset(self, *args, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment and update the observation space.

        Args: All arguments are forwarded to the env provided to __init__.

        Returns
        -------
        observation : dict[str, Any]
            The observations corresponding to each agent, translated into a vector format.
        info : dict[str, dict]
            Forwarded from self.env.
        """
        observations, info = super().reset(*args, **kwargs)
   
        obs = ({
            a: {"observations": observations[a], "action_mask": 1*np.array(self.action_mask(a))} for a in self.agents
        })
        assert isinstance(obs, dict)
        assert info.keys() == obs.keys()
        return obs, info
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent_name: str) -> Space | spaces.Dict:
        """Returns the multi-discrete space corresponding to the given agent."""
        return self._observation_space[agent_name]

    @functools.lru_cache(maxsize=None)
    def observation_spaces(self) -> dict[str, Space | spaces.Dict]:
        """Returns multi-discrete spaces corresponding to each agent."""
        return {a: self.observation_space(a) for a in self.possible_agents}

