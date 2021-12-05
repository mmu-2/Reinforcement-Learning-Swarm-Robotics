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


y_rewards = list()
y_average = list()
y_100 = list()
x_axis = list()
window = list()

last_100_average = 0
average_reward = 0
i_episode = 0

load_checkpoint = False #just manually set this

if load_checkpoint == True:
    #additional checkpoint data so I can reload and start from a later checkpoint
    file = open("./ac_training_data/window", "rb")
    window = pickle.load(file)
    file.close()
    file = open("./ac_training_data/data", "rb")
    [average_reward, i_episode] = pickle.load(file)
    file.close()
    #graph-related data            
    file = open("./ac_training_data/rewards", "rb")
    y_rewards = pickle.load(file)
    file.close()
    file = open("./ac_training_data/last100average", "rb")
    y_100 = pickle.load(file)
    file.close()
    file = open("./ac_training_data/running_average", "rb")
    y_average = pickle.load(file)
    file.close()
    file = open("./ac_training_data/episodes", "rb")
    x_axis = pickle.load(file)
    file.close()
    for i in range(len(agents)):
        agents[i].load('agent{}-checkpoint{}'.format(i+1,i_episode))

while last_100_average < 300:
    i_episode += 1
    episode_reward = 0
    observation = env.reset()
    rewards1 = 0
    rewards2 = 0
    rewards3 = 0
    rewards4 = 0
    done = False
    while done == False:
        # if i_episode % 100 == 0:
        #     for agent in agents:
        #         agent.debug = True
        #     env.render()
            # if last_100_average > 10:
            #     env.render()
        # if i_episode > 100:
        #     env.render()
        # env.render()

        obs1 = np.array(observation["agent1"])
        obs2 = np.array(observation["agent2"])
        obs3 = np.array(observation["agent3"])
        obs4 = np.array(observation["agent4"])
        target_obs = np.array(observation["target"])
        # The first obs must be the current agent. This way we can train each agent
        # the same to learn the first is the agent being controlled.
        # observation1 = np.concatenate((obs1, obs2, obs3, obs4, target_obs),axis=0)
        # observation2 = np.concatenate((obs2, obs1, obs3, obs4, target_obs),axis=0)
        # observation3 = np.concatenate((obs3, obs1, obs2, obs4, target_obs),axis=0)
        # observation4 = np.concatenate((obs4, obs1, obs2, obs3, target_obs),axis=0)
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
        # The first obs must be the current agent. This way we can train each agent
        # the same to learn the first is the agent being controlled.
        # observation1 = np.concatenate((obs1, obs2, obs3, obs4, target_obs),axis=0)
        # observation2 = np.concatenate((obs2, obs1, obs3, obs4, target_obs),axis=0)
        # observation3 = np.concatenate((obs3, obs1, obs2, obs4, target_obs),axis=0)
        # observation4 = np.concatenate((obs4, obs1, obs2, obs3, target_obs),axis=0)
        observation1 = np.concatenate((obs1, target_obs),axis=0)
        observation2 = np.concatenate((obs2, target_obs),axis=0)
        observation3 = np.concatenate((obs3, target_obs),axis=0)
        observation4 = np.concatenate((obs4, target_obs),axis=0)
        observations = [observation1, observation2, observation3, observation4]

        episode_reward += reward
        for i in range(len(agents)):
            agents[i].observe(last_observations[i], actions[i], rewards[i], observations[i], done)
            agents[i].debug = False
        if done:
            # print(rewards1)
            # print(rewards2)
            # print(rewards3)
            # print(rewards4)
            window.append(episode_reward)
            window = window[-100:]
            last_100_average = sum(window)/len(window)
            average_reward = average_reward + (episode_reward - average_reward)/(i_episode+1)
            if i_episode % 1 == 0:
                print("Episode {}: average: {:.3f} last 100: {:.3f} current reward: {:.3f}".format(i_episode, average_reward, last_100_average, episode_reward))
                for i in range(len(agents)):
                    agents[i].save('agent{}-checkpoint{}'.format(i+1,i_episode))
                    # agents[i].debug = True
                y_rewards.append(episode_reward)
                y_100.append(last_100_average)
                y_average.append(average_reward)
                x_axis.append(i_episode)

                #additional checkpoint data so I can reload and start from a later checkpoint
                file = open("./ac_training_data/window", "wb")
                pickle.dump(window, file)
                file.close()
                data = [average_reward, i_episode]
                file = open("./ac_training_data/data", "wb")
                pickle.dump(data, file)
                file.close()
                #graph-related data
                file = open("./ac_training_data/rewards", "wb")
                pickle.dump(y_rewards, file)
                file.close()
                file = open("./ac_training_data/last100average", "wb")
                pickle.dump(y_100, file)
                file.close()
                file = open("./ac_training_data/running_average", "wb")
                pickle.dump(y_average, file)
                file.close()
                file = open("./ac_training_data/episodes", "wb")
                pickle.dump(x_axis, file)
                file.close()
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
    agents[i].save('agent{}-checkpoint{}'.format(i+1,i_episode))
env.close()