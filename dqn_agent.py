import torch
import numpy as np
import random
import copy
from collections import deque
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

config = {
    'CAPACITY': 1000,
    'MINIBATCH': 250,
    "RESET_WEIGHT_COUNTER" : 100,
    "TRAIN_TIMESTEPS" : 3, # 10
    "GAMMA" : 0.99, # lower gamma
    "MINIMUM_EXPLORATION_PROBABILITY" : 0.01,
    "MAXIMUM_EXPLORATION_PROBABILITY" : 1,
    "EXPLORATION_PROBABILITY" : 1,
    "EPSILON_DECAY" : 0.005,
    "MAX_TIMESTEPS": 15,
    "NO_EPISODES_TRAINING": 7000,
    "LEARNING_RATE": 0.001
}


class dqn_nn(nn.Module):
    def __init__(self):
        super(dqn_nn, self).__init__()

        # Current agent, agent2, agent3, agent4, target

        # self.fc1_actor = nn.Linear(20, 256)
        self.fc1 = nn.Linear(8, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 5)  # nop, left, right, up, down

    def forward(self, x):
        pol = F.relu(self.fc1(x))
        pol = F.relu(self.fc2(pol))
        pol = F.relu(self.fc3(pol))
        pol = self.fc4(pol)
        return pol


class dqn_agent:
    def __init__(self, env=None):

        self.debug = False

        self.current_state = np.zeros(4)

        self.dqn = dqn_nn()

        # self.alpha = .001  # for now, 1 learning rate. May need 2 if doesn't converge
        # self.gamma = .99

        # self.ac_optimizer = optim.AdamW(self.ac.parameters(), lr=self.alpha)
        # # self.ac_optimizer = optim.SGD(self.ac.parameters(), lr=self.alpha)
        # self.ac_criterion = nn.MSELoss()

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.values = []
        self.probs = []
        self.logs = []

        self.env = env
        self.config = config
        # initalize replay memory
        self.replay_memory = deque(maxlen = self.config["CAPACITY"])
        # initialize q values
        self.q_values = dqn_nn()
        self.q_values_target = dqn_nn()
        self.q_values_target.load_state_dict(self.q_values.state_dict())
        self.learning_steps = 0
        self.exploration_probability = self.config["EXPLORATION_PROBABILITY"]
        self.minimum_exploration_probability = self.config["MINIMUM_EXPLORATION_PROBABILITY"]
        self.maximum_exploration_probability = self.config["MAXIMUM_EXPLORATION_PROBABILITY"]
        self.exploration_decay_rate = self.config["EPSILON_DECAY"]
        self.optimizer = optim.Adam(self.q_values.parameters(), lr=self.config["LEARNING_RATE"]) # 0.01

    def load(self, filename):
        self.dqn.load_state_dict(torch.load('./models/{}-dqn-weights.pth'.format(filename)))

    def save(self, filename):
        torch.save(self.dqn.state_dict(), './models/{}-dqn-weights.pth'.format(filename))

    def step(self, observation):
        # vals, policy_dist = self.ac(torch.from_numpy(observation))
        # # if self.debug == True:
        # #     print(vals)
        # #     print(policy_dist)
        # action = np.random.choice(5, p=policy_dist.detach().numpy())
        if np.random.uniform() > self.exploration_probability:
            actions = self.q_values(Variable(torch.unsqueeze(torch.FloatTensor(observation), 0)))
            action = torch.max(actions, 1)[1].data.numpy()[0]
        else:
            action = np.random.randint(5)
        return action

    def observe(self, last_observation, action, reward, observation, done, episode):

        self.states.append(last_observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(observation)
        self.dones.append(done)

        self.replay_memory.append((last_observation, action, reward, observation, done))
        if not len(self.dones) % self.config["TRAIN_TIMESTEPS"] and len(self.replay_memory) > self.config[
            "MINIBATCH"]:
            if not self.learning_steps % self.config["RESET_WEIGHT_COUNTER"]:
                self.q_values_target.load_state_dict(self.q_values.state_dict())
            training_batch = random.sample(self.replay_memory, self.config["MINIBATCH"])
            states_list, actions_list, rewards_list, next_state_list, done_list = (
                torch.FloatTensor(np.array(list(lst))) for lst in zip(*training_batch))
            q_values_selected, q_target = self.update_qtable(states_list, actions_list, rewards_list,
                                                             next_state_list, done_list)
            self.calculate_loss(q_values_selected, q_target)
            self.learning_steps += 1
        # value, probs = self.dqn(torch.from_numpy(last_observation))
        # # value = value.detach().numpy()
        # self.values.append(value)
        # self.logs.append(torch.log(probs[action]))

        if done == True:
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.next_states.clear()
            self.dones.clear()
            self.values.clear()
            self.probs.clear()
            self.logs.clear()
            # self.epsilon_decay_list.append(self.exploration_probability)
            self.exploration_probability = self.minimum_exploration_probability + (
                    self.maximum_exploration_probability - self.minimum_exploration_probability) * math.exp(
                -self.exploration_decay_rate * episode)

    def update_qtable(self, states_list, actions_list, rewards_list, next_state_list, done_list):
        q_values = self.q_values(states_list)
        q_values_next_state = self.q_values_target(next_state_list)
        q_values_selected = torch.Tensor(np.zeros(self.config["MINIBATCH"]))
        q_target = np.zeros(self.config["MINIBATCH"])
        for i in range(self.config["MINIBATCH"]):
            if done_list[i]:
                target_value = rewards_list[i]
            else:
                target_value = rewards_list[i] + self.config["GAMMA"] * q_values_next_state[i].max()
            q_values_selected[i] = q_values[i][int(actions_list[i])]
            q_target[i] = target_value
        q_target = torch.FloatTensor(q_target)
        return q_values_selected, q_target

    def calculate_loss(self, q_values_selected, q_target):
        loss = nn.MSELoss()(q_values_selected, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # print("Loss - ", loss)
        # self.loss_list.append(loss)




