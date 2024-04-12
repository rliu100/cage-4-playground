from CybORG import CybORG
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from CybORG.Agents.Wrappers import BlueFixedActionWrapper
from rich import print

sg = EnterpriseScenarioGenerator()
cyborg = CybORG(scenario_generator=sg)

env = BlueFixedActionWrapper(env=cyborg)
obs, _ = env.reset()

print(obs.keys())

