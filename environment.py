import numpy as np
import heapq
import gymnasium as gym
from gym import spaces

from utils.make_instances import generate_random_map, map_partition, generate_random_agents


class POMAPFEnv(gym.Env):
    def __init__(self, config, curriculum=True):
        self.curriculum = curriculum
        self.OBSTACLE, self.FREE_SPACE = config['grid_map']['OBSTACLE'], config['grid_map']['FREE_SPACE']
        self.fov_size = tuple(config['fov_size'])
        self.obs_r = (int(np.floor(self.fov_size[0]/2)), int(np.floor(self.fov_size[1]/2)))
        self.K_obs = config['K_obs']
        self.ind_reward_func = config['ind_reward_func']
        self.lambda_r = config['lambda_r']
        self.action_id = config['action_id']
        self.action_dim = config['action_dim']
        self.num_agents = config['default_num_agents']
        self.min_num_agents, self.max_num_agents = self.num_agents, self.num_agents
        self.map_size = config['default_map_size']
        self.min_map_size, self.max_map_size = self.map_size, self.map_size
        self.map_height, self.map_width = self.map_size, self.map_size
        self.grid_map = None
        self.state_n, self.goal_state_n = None, None
        self.load_random_map()
        self.heuristic_maps = {}
        self._get_heuristic_maps()

    @property
    def observation_space(self):
        return [spaces.Box(low=-np.inf, high=+np.inf, shape=(3 * self.K_obs, self.fov_size[0], self.fov_size[1])) for _ in range(self.num_agents)]

    @property
    def action_space(self):
        return [spaces.Discrete(self.action_dim) for _ in range(self.num_agents)]
    
    def get_spaces(self):
        return (self.observation_space, self.action_space)

    def seed(self, seed=0):
        np.random.seed(seed)

    def get_random_ranges(self):
        return self.min_num_agents, self.max_num_agents, self.min_map_size, self.max_map_size

    def set_random_ranges(self, min_num_agents, max_num_agents, min_map_size, max_map_size):
        self.min_num_agents, self.max_num_agents = min_num_agents, max_num_agents
        self.min_map_size, self.max_map_size = min_map_size, max_map_size

    def load_random_map(self):
        max_num_agents = self.max_num_agents if self.curriculum else self.min_num_agents
        max_map_size = self.max_map_size if self.curriculum else self.max_map_size
        self.num_agents = np.random.choice(range(self.min_num_agents, max_num_agents+1))
        self.map_height = np.random.choice(range(self.min_map_size, max_map_size+1, 5))
        self.map_width = np.random.choice(range(self.min_map_size, max_map_size+1, 5))
        num_obstacles = np.floor(self.map_height * self.map_width * np.random.triangular(0, 0.33, 0.5))
        self.grid_map = generate_random_map(self.map_height, self.map_width, num_obstacles)
        map_partitions = map_partition(self.grid_map)
        self.state_n, self.goal_state_n = generate_random_agents(self.grid_map, map_partitions, self.num_agents)

    def load(self, grid_map, state_n, goal_state_n):
        self.grid_map = grid_map
        self.state_n = state_n
        self.goal_state_n = goal_state_n        
        self.num_agents = len(self.state_n)
        self.map_height, self.map_width = len(self.grid_map), len(self.grid_map[0])
        self._get_heuristic_maps()

    def reset(self):
        self.load_random_map()
        return self.observe()[0]

    def _move(self, loc, id):
        action = self.action_id[id]
        return (loc[0] + action[0], loc[1] + action[1])

    def _get_heuristic_maps(self):
        self.heuristic_maps = {}
        self.padded_heuristic_maps = {}

        for i in range(self.num_agents):
            self.heuristic_maps[i] = np.zeros(np.shape(self.grid_map))
            goal = self.goal_state_n[i]
            open_list = []
            closed_list = {}
            root = {'loc': goal, 'cost': 0}
            heapq.heappush(open_list, (root['cost'], goal, root))
            closed_list[goal] = root
            while len(open_list) > 0:
                (cost, loc, curr) = heapq.heappop(open_list)
                for d in range(4):
                    child_loc = self._move(loc, d)
                    child_cost = cost + 1
                    if child_loc[0] < 0 or child_loc[0] >= self.map_height \
                    or child_loc[1] < 0 or child_loc[1] >= self.map_width:
                        continue
                    if self.grid_map[child_loc[0]][child_loc[1]] == self.OBSTACLE:
                        continue
                    child = {'loc': child_loc, 'cost': child_cost}
                    if child_loc in closed_list:
                        existing_node = closed_list[child_loc]
                        if existing_node['cost'] > child_cost:
                            closed_list[child_loc] = child
                            heapq.heappush(open_list, (child_cost, child_loc, child))
                    else:
                        closed_list[child_loc] = child
                        heapq.heappush(open_list, (child_cost, child_loc, child))

            for x in range(self.map_height):
                for y in range(self.map_width):
                    if (x, y) in closed_list:
                        self.heuristic_maps[i][x][y] = closed_list[(x, y)]['cost'] / (self.map_height * self.map_width)
                    else:
                        self.heuristic_maps[i][x][y] = 1.0

    def _detect_vertex_collision(self, path1, path2):
        if np.array_equal(path1[1], path2[1]):
            return True
        return False

    def _detect_edge_collision(self, path1, path2):
        if np.array_equal(path1[1], path2[0]) and np.array_equal(path1[0], path2[1]):
            return True
        return False

    def step(self, action_n):
        reward_n = [0 for _ in range(self.num_agents)]
        paths = []
        for i in range(self.num_agents):
            next_state = self.state_n[i]
            if action_n[i] == 4:
                if np.array_equal(next_state, self.goal_state_n[i]):
                    reward_n[i] = self.ind_reward_func['stay_on_goal']
                else:
                    reward_n[i] = self.ind_reward_func['stay_off_goal']
            else:
                x, y = self._move(self.state_n[i], action_n[i])
                # obstacle check
                if 0 <= x < self.map_height and 0 <= x < self.map_width and self.grid_map[x][y] == self.FREE_SPACE:
                    next_state = (x, y)
                    reward_n[i] = self.ind_reward_func['move'] - (1 - self.lambda_r) * self.heuristic_maps[i][x][y]
                else:
                    reward_n[i] = self.ind_reward_func['collision']
            paths.append([self.state_n[i], next_state])
        # edge collision check
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                if self._detect_edge_collision(paths[i], paths[j]):
                    paths[i][1] = self.state_n[i]
                    paths[j][1] = self.state_n[j]
                    reward_n[i] = self.ind_reward_func['collision']
                    reward_n[j] = self.ind_reward_func['collision']
        # vertex collision check
        collision_flag = True
        while collision_flag:
            collision_flag = False
            for i in range(self.num_agents):
                for j in range(i+1, self.num_agents):
                    if self._detect_vertex_collision(paths[i], paths[j]):
                        k = i
                        if np.array_equal(paths[i][0], paths[i][1]):
                            k = j
                        elif np.array_equal(paths[j][0], paths[j][1]):
                            k = i
                        else:
                            x, y = paths[i][1]
                            if self.heuristic_maps[i][x][y] > self.heuristic_maps[j][x][y]:
                                k = j
                        paths[k][1] = self.state_n[k]
                        reward_n[k] = self.ind_reward_func['collision']
                        collision_flag = True
                        break

        next_state_n = []
        for i in range(self.num_agents):
            next_state_n.append(paths[i][1])
        self.state_n = next_state_n
        done_n = np.equal(self.state_n, self.goal_state_n)
        if all(done_n):
            reward_n = [self.ind_reward_func['reach_goal'] for _ in range(self.num_agents)]
        obs_n, _ = self.observe()
        return obs_n, reward_n, done_n, done_n.sum()

    def observe(self):
        padded_grid_map = np.pad(self.grid_map, ((self.obs_r[0], self.obs_r[0]), (self.obs_r[1], self.obs_r[1])), mode='constant', constant_values=self.OBSTACLE)
        padded_heuristic_maps = {}
        for i in range(self.num_agents):
            padded_heuristic_maps[i] =  np.pad(self.heuristic_maps[i], ((self.obs_r[0], self.obs_r[0]), (self.obs_r[1], self.obs_r[1])), mode='constant', constant_values=1.0)
        observed_agents = {}
        for i in range(self.num_agents):
            p = []
            for j in range(self.num_agents):
                x_i, y_i = self.state_n[i][0], self.state_n[i][1]
                x_j, y_j = self.state_n[j][0], self.state_n[j][1]
                if abs(x_i - x_j) <= self.obs_r[0] \
                    and abs(y_i - y_j) <= self.obs_r[1]:
                    heapq.heappush(p, (abs(x_i - x_j) + abs(y_i - y_j), j))
            observed_agents[i] = []
            for k in range(self.K_obs):
                if len(p):
                    _, a = heapq.heappop(p)
                    observed_agents[i].append(a)
                else:
                    break
        obs_n = []
        for i in range(self.num_agents):
            obs_n.append(np.zeros((3 * self.K_obs, self.fov_size[0], self.fov_size[1]), dtype=float))
            for k, j in enumerate(observed_agents[i]):
                obs_n[i][3*k] = padded_grid_map[self.state_n[i][0]:self.state_n[i][0]+self.fov_size[0],self.state_n[i][1]:self.state_n[i][1]+self.fov_size[1]]
                o_x = self.obs_r[0] - self.state_n[i][0] + self.state_n[j][0]
                o_y = self.obs_r[1] - self.state_n[i][1] + self.state_n[j][1]
                obs_n[i][3*k+1][o_x][o_y] = 1.0
                obs_n[i][3*k+2] = padded_heuristic_maps[j][self.state_n[i][0]:self.state_n[i][0]+self.fov_size[0],self.state_n[i][1]:self.state_n[i][1]+self.fov_size[1]]
        return obs_n, self.state_n # obs_n[i]: (3 * self.K_obs, fov_size[0], fov_size[1])
    
    def upgrade(self):
        if self.max_map_size + 5 > self.config['max_map_size'] and self.max_num_agents + 2 > self.config['num_agents']:
            return False
        if self.max_map_size + 5 > self.config['max_map_size']:
            self.max_num_agents += 2
            return True
        if self.max_num_agents + 2 > self.config['num_agents']:
            self.max_map_size += 5
            return True            
        if np.random.uniform() < 0.5:
            self.max_map_size += 2
        else:
            self.max_map_size += 5
        return True