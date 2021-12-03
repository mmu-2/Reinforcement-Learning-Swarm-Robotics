from ac_agent import ac_agent
from working_env import Environment

import matplotlib.pyplot as plt
import torch
import numpy as np

EPISODES = 1000
N_AGENTS = 4 #DON'T CHANGE THIS

env = Environment()
agents = []
for i in range(N_AGENTS):
    agents.append(ac_agent())


y_rewards = list()
y_average = list()
y_100 = list()
x_axis = list()
window = list()

last_100_average = 0
average_reward = 0
i_episode = 0
while last_100_average < 250:
    i_episode += 1
    episode_reward = 0
    observation = env.reset()
    done = False
    while done == False:
        # if i_episode > EPISODES - 2:
            # env.render()
        # if i_episode > 200:
        #     env.render()
        # env.render()

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

        last_observations = [observation1, observation2, observation3, observation4]
        actions = []
        for i in range(len(agents)):
            actions.append(agents[i].step(last_observations[i]))

        observation, reward, done, info = env.step(actions)

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
        observations = [observation1, observation2, observation3, observation4]

        episode_reward += reward
        for i in range(len(agents)):
            agents[i].observe(last_observations[i], actions[i], reward, observations[i], done)
            agents[i].debug = False
        if done:
            window.append(episode_reward)
            window = window[-100:]
            last_100_average = sum(window)/len(window)
            average_reward = average_reward + (episode_reward - average_reward)/(i_episode+1)
            if i_episode % (EPISODES / 100) == 0:
                print("Episode {}: average: {:.3f} last 100: {:.3f} current reward: {:.3f}".format(i_episode, average_reward, last_100_average, episode_reward))
                for i in range(len(agents)):
                    agents[i].save('agent{}-checkpoint{}'.format(i,i_episode))
                    agents[i].debug = True
                y_rewards.append(episode_reward)
                y_100.append(last_100_average)
                y_average.append(average_reward)
                x_axis.append(i_episode)
            break

plt.title('Reward Per Episode')
plt.plot(x_axis, y_average, label='cumulative average reward')
plt.plot(x_axis, y_100, label='last 100 averaged reward')
plt.plot(x_axis, y_rewards, label='episode reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()

for i in range(len(agents)):
    agents[i].save('agent{}-checkpoint{}'.format(i,i_episode))
env.close()