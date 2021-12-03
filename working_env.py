#!/usr/bin/env python3
"""
Official first environment of Swarm Robotics Project

By default, will load default.xml, which consists of a body
in the center of a bounded 3x3 area. There are two agents located
in corners of the environment.

Useful references:
https://mujoco.readthedocs.io/en/latest/programming.html
https://mujoco.readthedocs.io/en/latest/APIreference.html
https://openai.github.io/mujoco-py/build/html/reference.html#pymjdata-time-dependent-data
"""
from mujoco_py import load_model_from_xml, load_model_from_path, MjSim, MjViewer
import os

import random
import numpy as np
import glfw #screen creator, often used with opengl/pyglet


#To see order or to change something with the movable parts of
#the agent, check xmls/ directory

#Actions right now:
#sim.data.ctrl[0] = X direction of bot1 [-5 to 5] determines speed
#sim.data.ctrl[1] = Y direction of bot1 [-5 to 5] determines speed
#sim.data.ctrl[2] = X direction of bot2 [-5 to 5] determines speed
#sim.data.ctrl[3] = Y direction of bot2 [-5 to 5] determines speed


"""
Description: A humanoid is dropped at position (0,0) in the center of the environment.
The environment is bounded to limit the environment to within 5x5 (meters? I think).

As we develop this environment, we may want to allow randomness and create custom XML so
that the agent is just learning to move to one specific spot.

Currently, the idea will be when the agents get close enough to the target, they will
automatically gain the target's information. In reality, this would require preprocessing.
However, implmentation specific details are not important.

Observation:
Position: No programmatic limit, but a physical barrier in default environment limits 
travel to between -5 and 5.
Velocity: Current bound is a combination of Force bounded to -10 to 10 and mass. This
will change with testing and I'm not currently sure on units and conversion to velocity.

Box() - i.e. continuous
[0]         Dictionary of each bot's name with their (x,y,xVelocity, yVelocity) positions.
[1]         Target information: None if the target hasn't been found. (x, y, xVelocity, yVelocity) if target has been found.

There is also Z information, but we don't really need that.
Note: We can add xquat and xvelr (rotation) information if we want as well.
The current information is only for the torso. We can pull all the other body
part information, but there are about 20 parts.

Actions:
Discrete()
A list of actions from bot1 ... to ... botN (currently bot4)
(0, 1, 2, 3, 4, 5)
(NOP, Left, Right, Up, Down)
For now, the bots can only travel at full speed.

e.g.
[[-1],[-1]
 [0],[0]
 [1],[1]
 [1],[1]]
 The shape of this is list[8][1].
 We can modify this if we want. I just did this for easy
 conversion to the data that MuJoCo likes.

Reward:
Reward +10 for discovering the target.
Reward is given based on the target's proximity to the destination.
The calculation is Euclidean distance. This means that moving the target
towards the destination provides a positive reward, but moving away provides
a negative reward.
Reward = old_target_destination_distance - new_target_destination_distance * 100
Reward +10 for reaching the destination.

Reward will be -1 every 1000 frame. This encourages the agent to progress.

Starting State:
The humanoid in a standing position. The agents will be preset to a starting
location. (The human will immediately fall). The agents will not have any
initialized random values. We may want to change this in the future to
better generalize our agents.

Episode Termination:
Episode length is greater than 20,000.
Target has reached destination.

Solved Requirements:
Considered solved when the average return is 500 over 100 trials.
(This is completely arbitrary right now, but we can test the feasibility
of this later.)

"""



class Environment:
    def __init__(self, name='default.xml'):
        self.model = load_model_from_path("xmls/"+name)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.reset_state = self.sim.get_state()
        self.SPEED = 5
        self.VISION_DIST = 1.5
        self.destination = [2, 2] #I just randomly chose this
        self.discovered = False
        self.timestep = 0
        self.MAX_TIMESTEP = 5000

        #timestep too short for random motion to learn
        #so I step TIME_INC times before next try
        self.TIME_INC = 30

        self.target_previous = np.float32(np.concatenate((self.sim.data.get_body_xpos("target")[:2],
                                self.sim.data.get_body_xvelp("target")[:2]),axis=0))

    #helper function in a class
    @staticmethod
    def dist(x, y): #x and y should be the same length
        return np.linalg.norm(x-y)

    def randomMoveInitialize(self):
        #I can't really randomize location, so I will randomize the first move
        for i in range(4):
            action = random.randint(0,4)
            if action == 0: #nop
                self.sim.data.ctrl[i*2] = 0
                self.sim.data.ctrl[i*2+1] = 0
            elif action == 1: #left
                self.sim.data.ctrl[i*2] = -1 * self.SPEED
                self.sim.data.ctrl[i*2+1] = 0
            elif action == 2: #right
                self.sim.data.ctrl[i*2] = 1 * self.SPEED
                self.sim.data.ctrl[i*2+1] = 0
            elif action == 3: #up
                self.sim.data.ctrl[i*2] = 0
                self.sim.data.ctrl[i*2+1] = 1 * self.SPEED
            elif action == 4: #down
                self.sim.data.ctrl[i*2] = 0
                self.sim.data.ctrl[i*2+1] = -1 * self.SPEED

        for i in range(1000):
            self.sim.step()


    def step(self, actions):
        # self.timestep += 1


        for i in range(len(actions)):
            action = actions[i]

            if action == 0: #nop
                self.sim.data.ctrl[i*2] = 0
                self.sim.data.ctrl[i*2+1] = 0
            elif action == 1: #left
                self.sim.data.ctrl[i*2] = -1 * self.SPEED
                self.sim.data.ctrl[i*2+1] = 0
            elif action == 2: #right
                self.sim.data.ctrl[i*2] = 1 * self.SPEED
                self.sim.data.ctrl[i*2+1] = 0
            elif action == 3: #up
                self.sim.data.ctrl[i*2] = 0
                self.sim.data.ctrl[i*2+1] = 1 * self.SPEED
            elif action == 4: #down
                self.sim.data.ctrl[i*2] = 0
                self.sim.data.ctrl[i*2+1] = -1 * self.SPEED

        for i in range(self.TIME_INC):
            self.sim.step()
            self.timestep+=1


        observation = {}
        target = np.concatenate((self.sim.data.get_body_xpos("target")[:2],
                                self.sim.data.get_body_xvelp("target")[:2]),axis=0)
        target = np.float32(target)

        agent1 = np.concatenate((self.sim.data.get_body_xpos("agent1")[:2],
                                self.sim.data.get_body_xvelp("agent1")[:2]),axis=0)
        agent1 = np.float32(agent1)
        observation["agent1"] = agent1

        agent2 = np.concatenate((self.sim.data.get_body_xpos("agent2")[:2],
                                self.sim.data.get_body_xvelp("agent2")[:2]),axis=0)
        agent2 = np.float32(agent2)
        observation["agent2"] = agent2

        agent3 = np.concatenate((self.sim.data.get_body_xpos("agent3")[:2],
                                self.sim.data.get_body_xvelp("agent3")[:2]),axis=0)
        agent3 = np.float32(agent3)
        observation["agent3"] = agent3

        agent4 = np.concatenate((self.sim.data.get_body_xpos("agent4")[:2],
                                self.sim.data.get_body_xvelp("agent4")[:2]),axis=0)
        agent4 = np.float32(agent4)
        observation["agent4"] = agent4

        agent1_dist = self.dist(agent1[:2], target[:2])
        agent2_dist = self.dist(agent2[:2], target[:2])
        agent3_dist = self.dist(agent3[:2], target[:2])
        agent4_dist = self.dist(agent4[:2], target[:2])

        target_to_dest = self.dist(target[:2], self.destination)

        reward = -.01

        

        if agent1_dist < self.VISION_DIST or agent2_dist < self.VISION_DIST or agent3_dist < self.VISION_DIST or agent4_dist < self.VISION_DIST:
            if self.discovered == False: # I have this condition so we reward finding the target the first time.
                self.discovered = True
            #make it so more agents close to object is better
            if agent1_dist < self.VISION_DIST:
                reward += .05
            if agent2_dist < self.VISION_DIST:
                reward += .05
            if agent3_dist < self.VISION_DIST:
                reward += .05
            if agent4_dist < self.VISION_DIST:
                reward += .05

        if self.discovered == True: # Once we've found the target once, we will always know its location.
            observation["target"] = target
            previous_target_to_dest = self.dist(self.target_previous[:2], self.destination)
            
            #when we get closer we get reward
            #when target_to dest gets further away, we get negative reward
            # print('previous_target_to_dest: {}'.format(previous_target_to_dest))
            # print('target_to_dest: {}'.format(target_to_dest))
            reward += (previous_target_to_dest - target_to_dest) * 100
        else:
            #Random out of bounds location.
            #Agent will have to learn that to ignore -100 -100 until the position is found.
            observation["target"] = np.array([-100, -100, 0, 0], dtype="float32")

        done = False

        if target_to_dest < 0.5:
            done = True
            reward += 100

        if(self.timestep > self.MAX_TIMESTEP):
            done = True

        self.target_previous = target

        return observation, reward, done, {} #If we need debug info we can put it here.
        
        
    def render(self):
        self.viewer.render()

    def reset(self):
        self.sim.set_state(self.reset_state)
        self.discovered = False
        self.timestep = 0

        self.randomMoveInitialize()

        observation = {}
        target = np.concatenate((self.sim.data.get_body_xpos("target")[:2],
                                self.sim.data.get_body_xvelp("target")[:2]),axis=0)
        target = np.float32(target)

        agent1 = np.concatenate((self.sim.data.get_body_xpos("agent1")[:2],
                                self.sim.data.get_body_xvelp("agent1")[:2]),axis=0)
        agent1 = np.float32(agent1)
        observation["agent1"] = agent1

        agent2 = np.concatenate((self.sim.data.get_body_xpos("agent2")[:2],
                                self.sim.data.get_body_xvelp("agent2")[:2]),axis=0)
        agent2 = np.float32(agent2)
        observation["agent2"] = agent2

        agent3 = np.concatenate((self.sim.data.get_body_xpos("agent3")[:2],
                                self.sim.data.get_body_xvelp("agent3")[:2]),axis=0)
        agent3 = np.float32(agent3)
        observation["agent3"] = agent3

        agent4 = np.concatenate((self.sim.data.get_body_xpos("agent4")[:2],
                                self.sim.data.get_body_xvelp("agent4")[:2]),axis=0)
        agent4 = np.float32(agent4)
        observation["agent4"] = agent4

        if self.discovered == True: # Once we've found the target once, we will always know its location.
            observation["target"] = target
        else:
            #Random out of bounds location.
            #Agent will have to learn that to ignore -100 -100 until the position is found.
            observation["target"] = np.array([-100, -100, 0, 0], dtype="float32")

        return observation


    def close(self):
        glfw.destroy_window(self.viewer.window)
