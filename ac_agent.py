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


class actor_critic_nn(nn.Module):
    def __init__(self):
        super(actor_critic_nn,self).__init__()

        #Current agent, agent2, agent3, agent4, target
        # self.fc1_critic = nn.Linear(20, 256)
        self.fc1_critic = nn.Linear(8, 64)
        self.fc2_critic = nn.Linear(64, 64)
        self.fc3_critic = nn.Linear(64, 64)
        self.fc4_critic = nn.Linear(64, 1)

        # self.fc1_actor = nn.Linear(20, 256)
        self.fc1_actor = nn.Linear(8, 64)
        self.fc2_actor = nn.Linear(64, 64)
        self.fc3_actor = nn.Linear(64, 64)
        self.fc4_actor = nn.Linear(64, 5) #nop, left, right, up, down

    def forward(self, x):
        val = F.relu(self.fc1_critic(x))
        val = F.relu(self.fc2_critic(val))
        val = F.relu(self.fc3_critic(val))
        val = self.fc4_critic(val)

        pol = F.relu(self.fc1_actor(x))
        pol = F.relu(self.fc2_actor(pol))
        pol = F.relu(self.fc3_actor(pol))
        pol = F.softmax(self.fc4_actor(pol), dim=0)

        return val, pol


class ac_agent:
    def __init__(self, env=None):

        self.debug = False

        self.current_state = np.zeros(4)

        self.ac = actor_critic_nn()

        self.alpha = .001 #for now, 1 learning rate. May need 2 if doesn't converge
        self.gamma = .99

        self.ac_optimizer = optim.AdamW(self.ac.parameters(), lr=self.alpha)
        # self.ac_optimizer = optim.SGD(self.ac.parameters(), lr=self.alpha)
        self.ac_criterion = nn.MSELoss()

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.values = []
        self.probs = []
        self.logs = []
        
    
    def load(self, filename):
        self.ac.load_state_dict(torch.load('./models/{}-ac-weights.pth'.format(filename)))
    
    def save(self, filename):
        torch.save(self.ac.state_dict(), './models/{}-ac-weights.pth'.format(filename))

    def step(self, observation):
        vals , policy_dist = self.ac(torch.from_numpy(observation))
        # if self.debug == True:
        #     print(vals)
        #     print(policy_dist)
        action = np.random.choice(5, p=policy_dist.detach().numpy())
        return action

    def observe(self, last_observation, action, reward, observation, done):

        self.states.append(last_observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(observation)
        self.dones.append(done)

        value, probs = self.ac(torch.from_numpy(last_observation))
        # value = value.detach().numpy()
        self.values.append(value)
        self.logs.append(torch.log(probs[action]))


        if done == True:
            self.update(last_observation, action, reward, observation, done)
            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.next_states.clear()
            self.dones.clear()
            self.values.clear()
            self.probs.clear()
            self.logs.clear()
        

    def update(self, last_observation, action, reward, observation, done):

        values = self.values
        values = torch.stack(values, dim=0)
        values = values.squeeze()
        
        # target, _ = self.ac(torch.from_numpy(observation))
        # target = target.detach().numpy()[0]
        target = torch.tensor(0).detach().numpy()
        targets = np.zeros_like(values.detach().numpy())
        for t in reversed(range(len(self.values))):
            target = self.rewards[t] + self.gamma * target
            targets[t] = target
        
        # values = [torch.from_numpy(v).float() for v in self.values]
        # values = torch.stack(values,dim=0)
        # values = values.squeeze()

        targets = torch.from_numpy(targets).squeeze().float()
        advantages = targets - values.detach()

        logs = torch.stack(self.logs)
        
        # actor_objective = (-logs * targets).mean()
        actor_objective = (-logs * advantages).mean()
        critic_loss = self.ac_criterion(targets, values)
        # print('t:{}'.format(targets))
        # print('v:{}'.format(values))
        # print('l:{}'.format(critic_loss))

        objective = actor_objective + critic_loss
        self.ac_optimizer.zero_grad()
        objective.backward()
        self.ac_optimizer.step()

