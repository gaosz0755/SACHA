import argparse
import pickle
from tqdm import tqdm
import numpy as np
import torch

from environment import POMAPFEnv
from model import AttentionPolicy

# config
import yaml
config = yaml.safe_load(open("./conf/config.yaml", 'r'))
eval_config = yaml.safe_load(open("./conf/eval_config.yaml", 'r'))
config = config | eval_config
OBSTACLE, FREE_SPACE = config['grid_map']['OBSTACLE'], config['grid_map']['FREE_SPACE']


def test_one_case(model, grid_map, starts, goals, max_timesteps):    
    env = POMAPFEnv(config)
    env.load(grid_map, starts, goals)
    obs_n, state_n = env.observe()
    step = 0
    num_agents = len(starts)
    paths = [[] for _ in range(num_agents)]
    while step <= max_timesteps:
        for i, loc in enumerate(env.state_n):
            paths[i].append(tuple(loc))
        action_n, _, _ = model(obs_n, state_n)
        obs_n, _, done_n, _ = env.step(action_n)
        for i, loc in enumerate(env.state_n):
            paths[i].append(tuple(loc))
        if all(done_n):
            break
        step += 1
    avg_step = 0.0
    for i in range(num_agents):
        while (len(paths[i]) > 1 and paths[i][-1] == paths[i][-2]):
            paths[i] = paths[i][:-1]
        avg_step += len(paths[i]) / num_agents
    return np.array_equal(env.state_n, env.goal_state_n), avg_step, paths


def main(args):
    state_dict = torch.load(args.load_from_dir)
    model = AttentionPolicy(args.communication)
    model.load_params(state_dict['policy'])
    num_instances = config['num_instances_per_test']
    for map_name, num_agents in config['test_settings']:
        file_name = f"./benchmarks/test_set/{map_name}_{num_agents}agents.pth"
        with open(file_name, 'rb') as f:
            instances = pickle.load(f)
        print(f"Testing instances for {map_name} with {num_agents} agents ...")
        success_rate, avg_step = 0.0, 0.0
        for grid_map, starts, goals in tqdm(instances[0: num_instances]):
            model.reset()
            done, steps, paths = test_one_case(model, np.array(grid_map), np.array(starts), np.array(goals), config['max_timesteps'][map_name])
            if done:
                success += 1 / num_instances
                avg_step += steps / num_instances
            else:
                avg_step += config['max_timesteps'][map_name] / num_instances
        with open(f"results.csv", 'a+') as f:
            height, width = np.shape(grid_map)
            num_obstacles = sum([row.count(OBSTACLE) for row in grid_map])
            method_name = 'SAHCA' if not args.communication else 'SACHA(C)'
            f.write(f"{method_name},{num_instances},{map_name},{height * width},{num_obstacles},{num_agents},{success_rate},{avg_step}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--communication", default=False, type=bool)
    parser.add_argument("--load_from_dir", default="")
    args = parser.parse_args()
    main(args)