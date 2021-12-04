# from ac_agent import ac_agent
from multiagent import multi_agent
from env import Environment

import matplotlib.pyplot as plt
import torch
import numpy as np

EPISODES = 100

env = Environment()
# agent1 = ac_agent()
# agent2 = ac_agent()
# agent3 = ac_agent()
# agent4 = ac_agent()

multiagent = multi_agent()


y_rewards = list()
y_average = list()
y_100 = list()
x_axis = list()
window = list()

last_100_average = 0
average_reward = 0
i_episode = -1
while last_100_average < 250:
    i_episode += 1
    episode_reward = 0
    observation = env.reset()
    done = False
    while done == False:
        # if i_episode > EPISODES - 2:
            # env.render()
        if i_episode > 50:
            env.render()

        #observation has everything
        actions = multiagent.step(observation)
        # I may need to convert numpy.ndarray to list
        last_observation = observation
        observation, reward, done, info = env.step(actions)
        episode_reward += reward
        multiagent.observe(last_observation, actions, reward, observation, done)
        multiagent.debug = False
        if done:
            window.append(episode_reward)
            window = window[-100:]
            last_100_average = sum(window)/len(window)
            average_reward = average_reward + (episode_reward - average_reward)/(i_episode+1)
            if i_episode % (EPISODES / 100) == 0:
                print("Episode {}: average: {:.3f} last 100: {} current reward: {}".format(i_episode, average_reward, last_100_average, episode_reward))
                multiagent.save('checkpoint{}'.format(i_episode))
                y_rewards.append(episode_reward)
                y_100.append(last_100_average)
                y_average.append(average_reward)
                x_axis.append(i_episode)
                multiagent.debug = True
            break
plt.title('Reward Per Episode')
plt.plot(x_axis, y_average, label='cumulative average reward')
plt.plot(x_axis, y_100, label='last 100 averaged reward')
plt.plot(x_axis, y_rewards, label='episode reward')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.show()


multiagent.save('final')
env.close()