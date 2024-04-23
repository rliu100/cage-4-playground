import sys
import time
import argparse

from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import EnterpriseMAE
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from Wrappers import MaskWrapper
from CustomRLLib import TorchActionMaskModel

from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models import ModelCatalog

NUM_AGENTS = 5
POLICY_MAP = {f"blue_agent_{i}": f"Agent{i}" for i in range(NUM_AGENTS)}

def policy_mapper(agent_id, episode, worker, **kwargs):
    return POLICY_MAP[agent_id]

def env_creator_CC4(env_config: dict, mask: bool):
    sg = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent,
        green_agent_class=EnterpriseGreenAgent,
        red_agent_class=FiniteStateRedAgent,
        steps=500
    )
    cyborg = CybORG(scenario_generator=sg)

    wrapper = MaskWrapper #if mask else EnterpriseMAE
    env = wrapper(env=cyborg, mask_training=mask)

    return env

def run_ppo(mask: bool = False, num_iters: int = 50):
    register_env(name='CC4', env_creator=lambda config: env_creator_CC4(config, mask))
    env = env_creator_CC4({}, mask)

    ModelCatalog.register_custom_model("torch_action_mask_model", TorchActionMaskModel)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    algo_config = (
        PPOConfig()
        .environment(env="CC4")
        .framework("torch")
        .debugging(logger_config={'logdir':f'./tb_logs/PPO_Example_{timestamp}', 'type': 'ray.tune.logger.TBXLogger'})
        .multi_agent(policies={
            ray_agent: PolicySpec(
                policy_class=None,
                observation_space=env.observation_space(cyborg_agent),
                action_space=env.action_space(cyborg_agent),
                config={'gamma': 0.85}
            ) for cyborg_agent, ray_agent in POLICY_MAP.items()
            },
            policy_mapping_fn=policy_mapper
        )
        .training(
            model={'custom_model': "torch_action_mask_model"}
        )
    )
    algo_config['evaluation_interval'] = 1
    algo_config['create_env_on_driver'] = True
    algo = algo_config.build()
    
    for i in range(num_iters):
        train_info=algo.train()
    
    filename = f'./Submissions/{"mask_" if mask else ""}results_{num_iters}_{timestamp}/staging/'
    algo.save(filename)

    output = algo.evaluate()
    print(output)
    print(
        "Avg episode length for trained agent: %.1f"
        % output["evaluation"]["episode_len_mean"]
    )
    return filename

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask', '-m', type=bool, default=False, help='use action masking')
    parser.add_argument('--num-iters', '-n', type=int, default=50, help='number of calls to algo.train()')
    arguments = parser.parse_args(argv)
    return arguments

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    filename = run_ppo(args.mask, args.num_iters)
    print(f"filename: {filename}")
