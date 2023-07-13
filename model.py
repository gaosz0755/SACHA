import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# config
import yaml
config = yaml.safe_load(open("./conf/config.yaml", 'r'))
policy_input_shape = (3 * config['K_obs'], config['fov_size'][0], config['fov_size'][1])
policy_output_shape = config['action_dim']
hidden_channels = config['hidden_channels']
hidden_dim = config['hidden_dim']
fov_size = tuple(config['fov_size'])
obs_r = (int(np.floor(fov_size[0]/2)), int(np.floor(fov_size[1]/2)))
num_heads = config['num_heads']
K_obs = config['K_obs']


class MHABlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.W_Q = nn.Linear(input_dim, output_dim * num_heads)
        self.W_K = nn.Linear(input_dim, output_dim * num_heads)
        self.W_V = nn.Linear(input_dim, output_dim * num_heads)
        self.W_O = nn.Linear(output_dim * num_heads, output_dim, bias=False)
        self.mha = nn.MultiheadAttention(output_dim * num_heads, num_heads, batch_first=True)

    def forward(self, x):
        output, _ = self.mha(x @ self.W_Q, x @ self.W_K, x @ self.W_V )
        output = output @ self.W_O
        return output


# https://github.com/tkipf/pygcn
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(torch.zeros(size=(in_features, out_features))))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(size=(out_features,))))
        else:
            self.register_parameter('bias', None)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        x = x @ self.weight
        if self.bias is not None:
            x += self.bias
        return torch.sparse.mm(adj, x)


class CommBlock(nn.Module):
    def __init__(self, hidden_dim, dropout=0.5, use_bias=True):
        super().__init__()
        self.gcn_1 = GCNLayer(hidden_dim, hidden_dim)
        self.gcn_2 = GCNLayer(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

    def initialize_weights(self):
        self.gcn_1.initialize_weights()
        self.gcn_2.initialize_weights()

    def forward(self, x, adj):
        x = F.relu(self.gcn_1(x, adj))
        x = self.dropout(x)
        x = self.gcn_2(x, adj)
        return x


class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1)

    def forward(self, x):
        output = x
        output = self.conv1(output)
        output = F.relu(output)
        output = self.conv2(output)
        output += x
        output = F.relu(output)
        return output


class AttentionPolicy(nn.Module):
    def __init__(self, communication, input_shape=policy_input_shape, output_shape=policy_output_shape,
                 hidden_channels=hidden_channels, hidden_dim=hidden_dim, num_heads=num_heads):
        super().__init__()
        self.communication = communication
        self.input_shape = policy_input_shape
        self.output_shape = policy_output_shape
        self.hidden_channels = hidden_channels
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            ResBlock(self.hidden_channels),
            ResBlock(self.hidden_channels),
            ResBlock(self.hidden_channels),
            nn.Conv2d(in_channels=hidden_channels, out_channels=16, kernel_size=1),
            nn.ReLU(True),
            nn.Flatten(),
        )
        self.memory_encoder = nn.GRUCell(16 * self.input_shape[0] * self.input_shape[1], self.hidden_dim)
        self.mha_block = MHABlock(self.hidden_dim, self.hidden_dim, self.num_heads)
        self.comm_block = CommBlock(self.hidden_dim)
        self.value_decoder = nn.Linear(self.hidden_dim, 1)
        self.advantage_decoder = nn.Linear(self.hidden_dim, self.output_shape)
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        self.hidden_state_n = None

    def _get_adj_from_states(self, state_n):
        num_agents = state_n.size(0)
        adj = np.zeros((num_agents, num_agents), dtype=float)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                x_i, y_i = state_n[i][0], state_n[i][1]
                x_j, y_j = state_n[j][0], state_n[j][1]
                if abs(x_i - x_j) <= obs_r[0] and abs(y_i - y_j) <= obs_r[1]:
                    adj[i][j] = 1.0
                    adj[j][i] = 1.0

    def forward(self, obs_n, state_n):
        num_agents = obs_n.size(0)
        for i in range(num_agents):
            o_i = [self.obs_encoder(obs_n[i][3*k:3*k+3]) for k in range(K_obs)]
            if not self.hidden_state_n:
                o_i = [self.memory_encoder(f_j) for f_j in o_i]
            else:
                o_i = [self.memory_encoder(f_j, self.hidden_state_n[i]) for f_j in o_i]
            o_i, _ = self.mha_bloack(o_i)
            self.hidden_state_n[i] = torch.sum(o_i, 1)
        if not self.communication:
            adj = torch.Tensor(self._get_adj_from_states(state_n))
            self.hidden_state_n = torch.Tensor(self.hidden_state_n)
            self.hidden_state_n = self.comm_block(self.hidden_state_n, adj).detach().numpy()
        V_n = [self.value_decoder(x) for x in self.hidden_state_n]
        A_n = [self.advantage_decoder(x) for x in self.hidden_state_n]
        Q_n = [V + A - A.mean(1, keepdim=True) for V, A in zip(V_n, A_n)]
        log_pi_n = F.log_softmax(Q_n, dim=1)
        action_n = [torch.argmax(Q, 1) for Q in Q_n]
        return action_n, self.hidden_state_n, log_pi_n
    
    def get_params(self):
        return {'policy': self.policy.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
    
    def reset(self):
        self.hidden_state_n = None


class AttentionCritic(nn.Module):
    def __init__(self, hidden_dim=hidden_dim, action_dim=policy_output_shape, num_heads=num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.obs_encoder = nn.Linear(self.hidden_dim, 16)
        self.concat_hidden_dim = hidden_dim + 16
        self.mha_block = MHABlock(self.concat_hidden_dim, self.concat_hidden_dim, self.num_heads)
        self.value_decoder = nn.Linear(self.concat_hidden_dim, 1)
        self.advantage_decoder = nn.Linear(self.concat_hidden_dim, self.action_dim)

    def _get_observed_agents_from_states(self, state_n):
        num_agents = state_n.size(0)
        observed_agents = [[] for _ in range(num_agents)]
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                x_i, y_i = state_n[i][0], state_n[i][1]
                x_j, y_j = state_n[j][0], state_n[j][1]
                if abs(x_i - x_j) <= obs_r[0] and abs(y_i - y_j) <= obs_r[1]:
                    observed_agents[i].append[j]
        return observed_agents

    def forward(self, hidden_state_n, action_n, state_n):
        num_agents = action_n.size(0)
        observed_agents = self._get_observed_agents_from_states(state_n)
        h_n = []
        for i in range(num_agents):
            c_i = []
            for j in observed_agents[i]:
                o_j = self.obs_encoder(hidden_state_n[i])
                a_j = action_n[i]
                c_i.append(torch.concat((o_j, a_j), 0))
            c_i, _ = self.mha_bloack(c_i)
            h_n.append(torch.sum(c_i, 1))
        V_n = [self.value_decoder(h) for h in h_n]
        A_n = [self.advantage_decoder(h) for h in h_n]
        Q_n = [V + A - A.mean(1, keepdim=True) for V, A in zip(V_n, A_n)]
        return Q_n
    
    def get_coma_baseline(self, hidden_state_n, action_n):
        b_n = []
        for i in range(len(action_n)):
            b = 0.0
            for j in range(self.action_dim):
                temp_action_n = copy.deepcopy(action_n)
                temp_action_n[i] = j
                b += self.forward(hidden_state_n, temp_action_n)[i] / self.action_dim
            b_n.append(b)
        return b_n

    def get_params(self):
        return {'critic': self.critic.state_dict(),
            'critic_optimizer': self.critic.state_dict()}

    def load_params(self, params):
        self.critic.load_state_dict(params['critic'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])


if __name__ == "__main__":
    policy = AttentionPolicy(False)