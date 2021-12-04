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
        self.fc1_critic = nn.Linear(20, 256)
        self.fc2_critic = nn.Linear(256, 256)
        self.fc3_critic = nn.Linear(256, 1)

        self.fc1_actor = nn.Linear(20, 256)
        self.fc2_actor = nn.Linear(256, 256)
        self.fc3_actor = nn.Linear(256, 5) #nop, left, right, up, down

    def forward(self, x):
        val = F.relu(self.fc1_critic(x))
        val = F.relu(self.fc2_critic(val))
        val = self.fc3_critic(val)

        pol = F.relu(self.fc1_actor(x))
        pol = F.relu(self.fc2_actor(pol))
        pol = F.softmax(self.fc3_actor(pol), dim=0)

        return val, pol


class multi_agent:
    def __init__(self, env=None):

        self.debug = False
        # self.env = env
        # self.observation_space = env.observation_space
        # self.action_space = env.action_space

        self.ac = actor_critic_nn()

        self.alpha = .001 #for now, 1 learning rate. May need 2 if doesn't converge
        self.gamma = .99

        self.ac_optimizer = optim.AdamW(self.ac.parameters(), lr=self.alpha)
        self.ac_criterion = nn.MSELoss()

        self.states1 = []
        self.actions1 = []
        self.next_states1 = []
        self.values1 = []
        self.probs1 = []
        self.logs1 = []

        self.states2 = []
        self.actions2 = []
        self.next_states2 = []
        self.values2 = []
        self.probs2 = []
        self.logs2 = []
        
        self.states3 = []
        self.actions3 = []
        self.next_states3 = []
        self.values3 = []
        self.probs3 = []
        self.logs3 = []

        self.states4 = []
        self.actions4 = []
        self.next_states4 = []
        self.values4 = []
        self.probs4 = []
        self.logs4 = []

        self.dones = []
        self.rewards = []
        
    
    def load(self, filename):
        self.ac.load_state_dict(torch.load('./models/{}-ac-weights.pth'.format(filename)))
    
    def save(self, filename):
        torch.save(self.ac.state_dict(), './models/{}-ac-weights.pth'.format(filename))

    def step(self, observation):


        obs1 = np.array(observation["agent1"])
        obs2 = np.array(observation["agent2"])
        obs3 = np.array(observation["agent3"])
        obs4 = np.array(observation["agent4"])
        target_obs = np.array(observation["target"])
        # The first obs must be the current agent. This way we can train each agent
        # the same to learn the first is the agent being controlled.
        observation1 = np.concatenate((obs1, obs2, obs3, obs4, target_obs),axis=0)
        observation2 = np.concatenate((obs2, obs1, obs3, obs4, target_obs),axis=0)
        observation3 = np.concatenate((obs3, obs1, obs2, obs4, target_obs),axis=0)
        observation4 = np.concatenate((obs4, obs1, obs2, obs3, target_obs),axis=0)

        _, policy_dist1 = self.ac(torch.from_numpy(observation1))
        _, policy_dist2 = self.ac(torch.from_numpy(observation2))
        _, policy_dist3 = self.ac(torch.from_numpy(observation3))
        _, policy_dist4 = self.ac(torch.from_numpy(observation4))

        if self.debug == True:
            print(policy_dist1)
            print(policy_dist2)
            print(policy_dist3)
            print(policy_dist4)

        #hardcoding 5 moves
        action1 = np.random.choice(5, p=policy_dist1.detach().numpy())
        action2 = np.random.choice(5, p=policy_dist2.detach().numpy())
        action3 = np.random.choice(5, p=policy_dist3.detach().numpy())
        action4 = np.random.choice(5, p=policy_dist4.detach().numpy())

        return [action1, action2, action3, action4]

    def observe(self, last_observation, actions, reward, observation, done):
        
        obs1 = np.array(last_observation["agent1"])
        obs2 = np.array(last_observation["agent2"])
        obs3 = np.array(last_observation["agent3"])
        obs4 = np.array(last_observation["agent4"])
        target_obs = np.array(last_observation["target"])
        # The first obs must be the current agent. This way we can train each agent
        # the same to learn the first is the agent being controlled.
        observation1 = np.concatenate((obs1, obs2, obs3, obs4, target_obs),axis=0)
        observation2 = np.concatenate((obs2, obs1, obs3, obs4, target_obs),axis=0)
        observation3 = np.concatenate((obs3, obs1, obs2, obs4, target_obs),axis=0)
        observation4 = np.concatenate((obs4, obs1, obs2, obs3, target_obs),axis=0)

        self.states1.append(observation1)
        self.states2.append(observation2)
        self.states3.append(observation3)
        self.states4.append(observation4)

        value, probs = self.ac(torch.from_numpy(observation1))
        value = value.detach().numpy()
        self.values1.append(value)
        self.logs1.append(torch.log(probs[actions[0]]))

        value, probs = self.ac(torch.from_numpy(observation2))
        value = value.detach().numpy()
        self.values2.append(value)
        self.logs2.append(torch.log(probs[actions[1]]))
        
        value, probs = self.ac(torch.from_numpy(observation3))
        value = value.detach().numpy()
        self.values3.append(value)
        self.logs3.append(torch.log(probs[actions[2]]))

        value, probs = self.ac(torch.from_numpy(observation4))
        value = value.detach().numpy()
        self.values4.append(value)
        self.logs4.append(torch.log(probs[actions[3]]))


        self.actions1.append(actions[0])
        self.actions2.append(actions[1])
        self.actions3.append(actions[2])
        self.actions4.append(actions[3])

        self.rewards.append(reward)

        obs1 = np.array(observation["agent1"])
        obs2 = np.array(observation["agent2"])
        obs3 = np.array(observation["agent3"])
        obs4 = np.array(observation["agent4"])
        target_obs = np.array(observation["target"])
        # The first obs must be the current agent. This way we can train each agent
        # the same to learn the first is the agent being controlled.
        observation1 = np.concatenate((obs1, obs2, obs3, obs4, target_obs),axis=0)
        observation2 = np.concatenate((obs2, obs1, obs3, obs4, target_obs),axis=0)
        observation3 = np.concatenate((obs3, obs1, obs2, obs4, target_obs),axis=0)
        observation4 = np.concatenate((obs4, obs1, obs2, obs3, target_obs),axis=0)
        
        self.next_states1.append(observation1)
        self.next_states2.append(observation2)
        self.next_states3.append(observation3)
        self.next_states4.append(observation4)

        self.dones.append(done)

        if done == True:
            self.update(last_observation, actions, reward, observation, done)
            self.states1.clear()
            self.states2.clear()
            self.states3.clear()
            self.states4.clear()
            self.actions1.clear()
            self.actions2.clear()
            self.actions3.clear()
            self.actions4.clear()
            self.rewards.clear()
            self.next_states1.clear()
            self.next_states2.clear()
            self.next_states3.clear()
            self.next_states4.clear()
            self.dones.clear()
            self.values1.clear()
            self.values2.clear()
            self.values3.clear()
            self.values4.clear()
            self.probs1.clear()
            self.probs2.clear()
            self.probs3.clear()
            self.probs4.clear()
            self.logs1.clear()
            self.logs2.clear()
            self.logs3.clear()
            self.logs4.clear()
        

    def update(self, last_observation, action, reward, observation, done):
        
        obs1 = np.array(last_observation["agent1"])
        obs2 = np.array(last_observation["agent2"])
        obs3 = np.array(last_observation["agent3"])
        obs4 = np.array(last_observation["agent4"])
        target_obs = np.array(last_observation["target"])
        # The first obs must be the current agent. This way we can train each agent
        # the same to learn the first is the agent being controlled.
        observation1 = np.concatenate((obs1, obs2, obs3, obs4, target_obs),axis=0)
        observation2 = np.concatenate((obs2, obs1, obs3, obs4, target_obs),axis=0)
        observation3 = np.concatenate((obs3, obs1, obs2, obs4, target_obs),axis=0)
        observation4 = np.concatenate((obs4, obs1, obs2, obs3, target_obs),axis=0)

        rand = random.randint(0,3)
        
        if rand == 0:
            target, _ = self.ac(torch.from_numpy(observation1))#observation not last_observ
            target = target.detach().numpy()[0]
            targets = np.zeros_like(self.values1)
            for t in reversed(range(len(self.values1))):
                target = self.rewards[t] + self.gamma * target
                targets[t] = target
            values = [torch.from_numpy(v).float() for v in self.values1]
            values = torch.stack(values,dim=0)
            values = values.squeeze()
            targets = torch.from_numpy(targets).squeeze().float()
            logs = torch.stack(self.logs1)
            actor_objective = (-logs * targets).mean()
            critic_loss = self.ac_criterion(targets, values)
            objective = actor_objective + critic_loss
            self.ac_optimizer.zero_grad()
            objective.backward()
            self.ac_optimizer.step()
        elif rand == 1:
            target, _ = self.ac(torch.from_numpy(observation2))
            target = target.detach().numpy()[0]
            targets = np.zeros_like(self.values2)
            for t in reversed(range(len(self.values2))):
                target = self.rewards[t] + self.gamma * target
                targets[t] = target
            values = [torch.from_numpy(v).float() for v in self.values2]
            values = torch.stack(values,dim=0)
            values = values.squeeze()
            targets = torch.from_numpy(targets).squeeze().float()
            logs = torch.stack(self.logs2)
            actor_objective = (-logs * targets).mean()
            critic_loss = self.ac_criterion(targets, values)
            objective = actor_objective + critic_loss
            self.ac_optimizer.zero_grad()
            objective.backward()
            self.ac_optimizer.step()
        elif rand == 2:
            target, _ = self.ac(torch.from_numpy(observation3))
            target = target.detach().numpy()[0]
            targets = np.zeros_like(self.values3)
            for t in reversed(range(len(self.values3))):
                target = self.rewards[t] + self.gamma * target
                targets[t] = target
            values = [torch.from_numpy(v).float() for v in self.values3]
            values = torch.stack(values,dim=0)
            values = values.squeeze()
            targets = torch.from_numpy(targets).squeeze().float()
            logs = torch.stack(self.logs3)
            actor_objective = (-logs * targets).mean()
            critic_loss = self.ac_criterion(targets, values)
            objective = actor_objective + critic_loss
            self.ac_optimizer.zero_grad()
            objective.backward()
            self.ac_optimizer.step()
        elif rand == 3:
            target, _ = self.ac(torch.from_numpy(observation4))
            target = target.detach().numpy()[0]
            targets = np.zeros_like(self.values4)
            for t in reversed(range(len(self.values4))):
                target = self.rewards[t] + self.gamma * target
                targets[t] = target
            values = [torch.from_numpy(v).float() for v in self.values4]
            values = torch.stack(values,dim=0)
            values = values.squeeze()
            targets = torch.from_numpy(targets).squeeze().float()
            logs = torch.stack(self.logs4)
            actor_objective = (-logs * targets).mean()
            critic_loss = self.ac_criterion(targets, values)
            objective = actor_objective + critic_loss
            self.ac_optimizer.zero_grad()
            objective.backward()
            self.ac_optimizer.step()
        else:
            print('I got the random range 0-3 wrong.')
