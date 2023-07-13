import copy
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn import MSELoss

from utils.misc import soft_update, hard_update, enable_gradients, disable_gradients
from model import AttentionPolicy, AttentionCritic


class SACHA:
    def __init__(self, config, communication, device):
        self.test = 0
        self.device = device
        self.policy = AttentionPolicy(communication)
        self.target_policy = copy.deepcopy(self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=config['lr_pi'])
        self.critic = AttentionCritic()
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config['lr_q'])
        self.tau = config['tau']
        self.gamma = config['gamma']
        self.alpha = config['alpha']
    
    def step(self, obs_n):
        return self.policy(obs_n)

    def update_policy(self, sample, soft=True):
        obs_n, hidden_n, action_n, reward_n, next_obs_n, done_n = sample
        curr_action_n, _, curr_log_pi_n = self.policy(obs_n, hidden_n)
        q_n = self.critic(hidden_n, action_n)
        b_n = self.critic.get_coma_baseline(hidden_n, action_n)
        loss = 0.0
        for i in range(len(obs_n)):
            log_pi = curr_log_pi_n[i]
            if soft:
                loss += (log_pi * (log_pi * self.alpha - q_n[i] + b_n[i])).detach().mean()
            else:
                loss += (log_pi * (- q_n[i] + b_n[i])).detach().mean()
        disable_gradients(self.critic)
        loss.backward()
        enable_gradients(self.critic)
        torch.nn.utils.clip_grad_norm(self.policy.parameters(), 0.5)
        self.policy.step()
        self.policy.zero_grad()  

    def update_critic(self, sample, soft=True):
        obs_n, hidden_n, action_n, reward_n, next_obs_n, done_n = sample
        next_action_n, next_hidden_n, next_log_pi_n = self.target_policy(next_obs_n, hidden_n)
        q_n = self.critic(action_n, hidden_n)
        next_q_n = self.target_critic(next_action_n, next_hidden_n)
        loss = 0.0
        for i, next_q, next_log_pi, q in zip(range(self.nagents), next_q_n, next_log_pi_n, q_n):
            target_q = (reward_n[i].view(-1, 1) + self.gamma * next_q * (1 - done_n[i].view(-1, 1)))
            if soft:
                target_q -= self.alpha * next_log_pi
            loss += MSELoss(q, target_q.detach())
        loss.backward()
        self.critic.scale_shared_grads()
        torch.nn.utils.clip_grad_norm(self.critic.parameters(), 10 * obs_n.size(0))
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()
        self.num_iterations += 1

    def update_targets(self):
        soft_update(self.target_policy, self.policy, self.tau)
        soft_update(self.target_critic, self.critic, self.tau)
    
    def prep_training(self, device):
        self.policy.train()
        self.critic.train()
        if device != self.device:
            fn = lambda x: x.cuda() if device == 'gpu' else lambda x: x.cpu()
            self.policy = fn(self.policy)
            self.critic = fn(self.critic)
            self.device = device

    def prep_rollout(self, device):
        self.policy.eval()
        if device != self.device:
            fn = lambda x: x.cuda() if device == 'gpu' else lambda x: x.cpu()
            self.policy = fn(self.policy)
            self.critic = fn(self.critic)

    def save(self, filename):
        state_dict = {
            'policy' : {
                'policy': self.policy.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict()
            },
            'critic' : {
                'critic': self.critic.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()
            }
        }
        torch.save(state_dict, filename)
