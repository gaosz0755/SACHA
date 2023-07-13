import argparse
import os
import numpy as np
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from environment import POMAPFEnv
from utils.replay_buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.sacha import SACHA

# device (CUDA_VISIBLE_DEVICES=GPU_ID)
device = 'gpu' if torch.cuda.is_available() else 'cpu'

# config
import yaml
config = yaml.safe_load(open("./conf/config.yaml", 'r'))
train_config = yaml.safe_load(open("./conf/train_config.yaml", 'r'))
config = config | train_config
OBSTACLE, FREE_SPACE = config['grid_map']['OBSTACLE'], config['grid_map']['FREE_SPACE']


def make_parallel_env(num_rollout_threads):
    def get_env_fn(rank):
        def init_env():
            env = POMAPFEnv(config)
            env.seed(rank)
            return env
        return init_env
    if num_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(num_rollout_threads)])


def main(args):
    log_dir = './log/SACHA' if not args.communication else './log/SACHA(C)'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))
    env = make_parallel_env(config['num_rollout_threads'])
    model = SACHA(config, args.communication, device)
    replay_buffer = ReplayBuffer(capacity=config['buffer_capacity'])
    num_steps = 0
    for ep_i in range(0, config['num_episodes'], config['num_rollout_threads']):
        obs = env.reset()
        model.prep_rollout('cpu')
        torch_obs = Variable(torch.Tensor(obs), requires_grad=False)
        for t_i in range(config['max_episode_length']):
            actions, hiddens, _ = model.step(torch_obs).data.numpy()
            next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, hiddens, actions, rewards, next_obs, dones)
            num_steps += config['num_rollout_threads']
            if len(replay_buffer) >= config['update_threshold'] and num_steps % config['steps_per_update'] < config['num_rollout_threads']:
                model.prep_training(device)
                for _ in range(config['updates_per_time']):
                    samples = replay_buffer.sample(config['batch_size'])
                    model.update_policy(samples)
                    model.update_critic(samples)
                    model.update_targets()
                model.prep_rollout('cpu')
        if replay_buffer.get_success_rate() > config['upgrade_threshold']:
            env.upgrade()
        logger.add_scalar('data/scalar1', replay_buffer.get_average_reward(), ep_i)
        if ep_i % config['save_interval'] < config['num_rollout_threads']:
            model.prep_rollout('cpu')            
            model.save(os.path.join(log_dir, f"model_ep{ep_i + 1}"))
        env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--communication", default=False, type=bool)
    args = parser.parse_args()
    main(args)