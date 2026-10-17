"""Read-only audit of the real official loader in a separate CPU process."""

import argparse
import importlib.metadata
from pathlib import Path

import d4rl
import gym
import numpy as np

from scripts.cql_antmaze_official_5090.common import read, verify, write

def audit(root):
    plan = read(root / "plan.json")
    verify(root, plan)
    package = root / "dependencies/d4rl/d4rl"
    if Path(d4rl.__file__).resolve() != package / "__init__.py":
        raise ValueError("Expected isolated, pinned official D4RL package")
    result = dict(d4rl_module=d4rl.__file__, versions={
        name: importlib.metadata.version(name)
        for name in ["torch", "gym", "numpy", "pyrallis", "wandb", "mujoco-py"]
    }, jobs={})
    for job in plan["jobs"]:
        env = gym.make(job["env"])
        raw = env.get_dataset()
        actual = d4rl.qlearning_dataset(env, dataset=raw)
        index = np.flatnonzero(~raw["timeouts"][:-1].astype(bool))
        for key in ["observations", "actions", "rewards", "terminals"]:
            np.testing.assert_array_equal(actual[key], raw[key][index])
        np.testing.assert_array_equal(actual["next_observations"],
                                      raw["observations"][index + 1])
        result["jobs"][job["id"]] = dict(
            transitions=len(index), success_terminals=int(actual["terminals"].sum()),
            timeout_transitions_dropped=int(raw["timeouts"][:-1].sum()),
            reward_values=np.unique(actual["rewards"]).tolist(),
            observations=env.observation_space.shape, actions=env.action_space.shape,
            full_array_comparison_passed=True,
        )
        env.reset()
        env.close()
        del raw, actual
    write(root / "data_audit.json", result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    audit(parser.parse_args().root.resolve())
