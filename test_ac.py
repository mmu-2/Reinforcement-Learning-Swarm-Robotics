from ac_agent import ac_agent
from working_env import Environment

import matplotlib.pyplot as plt
import torch
import numpy as np

import pickle

# EPISODES = 100
N_AGENTS = 4 #DON'T CHANGE THIS

env = Environment()
agents = []
for i in range(N_AGENTS):
    agents.append(ac_agent())

for i in range(len(agents)):
    # agents[i].load('best-agent{}'.format(i+1))
    agents[i].load('agent{}-checkpoint14041'.format(i+1))

y_rewards = list()
y_average = list()
y_100 = list()
x_axis = list()
window = list()

last_100_average = 0
average_reward = 0
i_episode = 0
while i_episode <= 9:
    i_episode += 1
    episode_reward = 0
    observation = env.reset()
    rewards1 = 0
    rewards2 = 0
    rewards3 = 0
    rewards4 = 0
    done = False
    while done == False:
        env.render()

        obs1 = np.array(observation["agent1"])
        obs2 = np.array(observation["agent2"])
        obs3 = np.array(observation["agent3"])
        obs4 = np.array(observation["agent4"])
        target_obs = np.array(observation["target"])
        observation1 = np.concatenate((obs1, target_obs),axis=0)
        observation2 = np.concatenate((obs2, target_obs),axis=0)
        observation3 = np.concatenate((obs3, target_obs),axis=0)
        observation4 = np.concatenate((obs4, target_obs),axis=0)

        last_observations = [observation1, observation2, observation3, observation4]
        actions = []
        for i in range(len(agents)):
            actions.append(agents[i].step(last_observations[i]))

        observation, rewards, done, info = env.step(actions)
        reward = sum(rewards)/len(rewards) #average of the agent rewards since there are multiple
        rewards1 += rewards[0]
        rewards2 += rewards[1]
        rewards3 += rewards[2]
        rewards4 += rewards[3]

        # if i_episode % 100 == 0:
        #     dist = env.dist(env.target_previous[:2], env.destination)
        #     print('{:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f}'.format(rewards1, rewards2, rewards3, rewards4, dist))

        obs1 = np.array(observation["agent1"])
        obs2 = np.array(observation["agent2"])
        obs3 = np.array(observation["agent3"])
        obs4 = np.array(observation["agent4"])
        target_obs = np.array(observation["target"])
        observation1 = np.concatenate((obs1, target_obs),axis=0)
        observation2 = np.concatenate((obs2, target_obs),axis=0)
        observation3 = np.concatenate((obs3, target_obs),axis=0)
        observation4 = np.concatenate((obs4, target_obs),axis=0)
        observations = [observation1, observation2, observation3, observation4]

        episode_reward += reward
        if done:
            window.append(episode_reward)
            window = window[-100:]
            last_100_average = sum(window)/len(window)
            average_reward = average_reward + (episode_reward - average_reward)/(i_episode+1)
            if i_episode % 1 == 0:
                print("Episode {}: average: {:.3f} last 100: {:.3f} current reward: {:.3f}".format(i_episode, average_reward, last_100_average, episode_reward))
                y_rewards.append(episode_reward)
                y_100.append(last_100_average)
                y_average.append(average_reward)
                x_axis.append(i_episode)
                
                file = open("./ac_testing_data/rewards", "wb")
                pickle.dump(y_rewards, file)
                file.close()
                # file = open("./ac_testing_data/last100average", "wb")
                # pickle.dump(y_100, file)
                # file.close()
                # file = open("./ac_testing_data/running_average", "wb")
                # pickle.dump(y_average, file)
                # file.close()
                file = open("./ac_testing_data/episodes", "wb")
                pickle.dump(x_axis, file)
                file.close()
            break

plt.title('Reward Per Episode')
plt.plot(x_axis, y_rewards, label='episode reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()