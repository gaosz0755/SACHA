import numpy as np
from torch import Tensor
from torch.autograd import Variable

# config
import yaml
config = yaml.safe_load(open("./conf/config.yaml", 'r'))
obs_shape = (3 * config['K_obs'], config['fov_size'][0], config['fov_size'][1])
hidden_dim = config['hidden_dim']
max_num_agents = config['max_num_agents']


class ReplayBuffer:
    def __init__(self, capacity, num_agents=max_num_agents, obs_shape=obs_shape, hidden_dim=hidden_dim):
        self.capacity = capacity
        self.num_agents = num_agents
        self.obs_buffs = []
        self.hidden_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        for i in range(max_num_agents):
            self.obs_buffs.append(np.zeros((capacity, *obs_shape), dtype=np.float32))
            self.hidden_buffs.append(np.zeros((capacity, hidden_dim), dtype=np.float32))
            self.ac_buffs.append(np.zeros((capacity), dtype=np.float32))
            self.rew_buffs.append(np.zeros(capacity, dtype=np.float32))
            self.next_obs_buffs.append(np.zeros((capacity, *obs_shape), dtype=np.float32))
            self.done_buffs.append(np.zeros(capacity, dtype=np.uint8))
        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, hiddens, actions, rewards, next_observations, dones):
        nentries = observations.shape[0]  # handle multiple parallel environments
        if self.curr_i + nentries > self.capacity:
            rollover = self.capacity - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(self.obs_buffs[agent_i],
                                                  rollover, axis=0)
                self.hidden_buffs[agent_i] = np.roll(self.hidden_buffs[agent_i],
                                                  rollover, axis=0)
                self.ac_buffs[agent_i] = np.roll(self.ac_buffs[agent_i],
                                                 rollover, axis=0)
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i],
                                                  rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i],
                                                   rollover)
            self.curr_i = 0
            self.filled_i = self.capacity
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(observations[:, agent_i])
            self.hidden_buffs[agent_i][self.curr_i:self.curr_i + nentries] = hiddens[:, agent_i]
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries] = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries] = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = np.vstack(next_observations[:, agent_i])
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dones[:, agent_i]
        self.curr_i += nentries
        if self.filled_i < self.capacity:
            self.filled_i += nentries
        if self.curr_i == self.capacity:
            self.curr_i = 0

    def sample(self, N, device, norm_rews=True):
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=True)
        if device:
            fn = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            fn = lambda x: Variable(Tensor(x), requires_grad=False)
        return ([fn(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [fn(self.hidden_buffs[i][inds]) for i in range(self.num_agents)],
                [fn(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                [fn(self.rew_buffs[i][inds]) for i in range(self.num_agents)],
                [fn(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [fn(self.done_buffs[i][inds]) for i in range(self.num_agents)])

    def get_success_rate(self, N):
        inds = np.arange(self.curr_i - N, self.curr_i) if self.filled_i == self.capacity \
            else np.arange(max(0, self.curr_i - N), self.curr_i)
        success_rate = 0.0
        for i in inds:
            if all(self.done_buffs[:,i]):
                success_rate += 1 / len(inds)
        return success_rate
    
    def get_average_reward(self, N):
        inds = np.arange(self.curr_i - N, self.curr_i) if self.filled_i == self.capacity \
            else np.arange(max(0, self.curr_i - N), self.curr_i)
        avg_rew = 0.0
        for i in inds:
            if all(self.done_buffs[:,i]):
                avg_rew += 1 / len(inds)
        return avg_rew