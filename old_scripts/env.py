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

import random #used for random placement of objects
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

def gen_xml():
    x = random.randint(-1, 1)
    y = random.randint(-1, 1)
    possible_starts = [(-4,-4),(-4,-3),(-4, -2),(-4, -1),(-4, 0),(-4, 1),(-4, 2),(-4, 3),(-4, 4),
                        (-3,-4),(-3,-3),(-3, -2),(-3, -1),(-3, 0),(-3, 1),(-3, 2),(-3, 3),(-3, 4),
                        (-2,-4),(-2,-3),(-2, -2),(-2, -1),(-2, 0),(-2, 1),(-2, 2),(-2, 3),(-2, 4),
                        (-1,-4),(-1,-3),(-1, -2),                         (1, 2),(1, 3),(1, 4),
                        (0,-4),(0,-3),(0, -2),                      (0, 2),(0, 3),(0, 4),
                        (1,-4),(1,-3),(1, -2),                      (1, 2),(1, 3),(1, 4),
                        (2,-4),(2,-3),(2, -2),(2, -1),(2, 0),(2, 1),(2, 2),(2, 3),(2, 4),
                        (3,-4),(3,-3),(3, -2),(3, -1),(3, 0),(3, 1),(3, 2),(3, 3),(3, 4),
                        (4,-4),(4,-3),(4, -2),(4, -1),(4, 0),(4, 1),(4, 2),(4, 3),(4, 4)]
    bots = random.sample(possible_starts,k=4)

    MODEL_XML = """

    <mujoco model="Swarm">

        <compiler angle="degree"/>

        <option timestep="0.01" iterations="50" solver="Newton" jacobian="sparse" cone="pyramidal" tolerance="1e-10"/>

        <size njmax="1500" nconmax="500" nstack="5000000"/>

        <default>
            <geom solimp=".9 .9 .01"/>

            <default class="humanoid">
                <geom material="humanoid"/>
                <joint damping="1" limited="true"/>
            </default>

            <default class="bot">
                <geom type="cylinder" material="bot" size="0.1 0.1" condim="4" mass="5" friction="1 .01 .01"/>
            </default>

            <default class="border">
                <geom type="capsule" size="0.4" rgba=".4 .4 .4 1"/>
            </default>

            <default class="borderpost">
                <geom type="box" size="0.41 0.41 0.41" rgba=".55 .55 .55 1"/>
            </default>

            <default class="position">
                <position ctrllimited="true" forcelimited="false"></position>
            </default>

            <joint limited="true" damping="1" armature="0"/>
            <motor ctrlrange="-10 10" ctrllimited="true"/>
        </default>

        <asset>
            <texture type="skybox" builtin="gradient" width="128" height="128" rgb1=".4 .6 .8" rgb2="0 0 0"/>
            <texture name="texgeom" type="cube" builtin="flat" mark="cross" width="128" height="128"
                rgb1="0.6 0.6 0.6" rgb2="0.6 0.6 0.6" markrgb="1 1 1"/>
            <texture name="texplane" type="2d" builtin="checker" rgb1=".4 .4 .4" rgb2=".6 .6 .6"
                width="512" height="512"/>
            <material name='MatPlane' reflectance='0.3' texture="texplane" texrepeat="1 1" texuniform="true"/>
            <material name='humanoid' texture="texgeom" texuniform="true" rgba="1.2 1.2 0.6 1"/>
            <material name='bot' texture="texgeom" texuniform="true" rgba=".8 .6 .8 1" />
        </asset>

        <visual>
            <quality shadowsize="4096" offsamples="8"/>
            <map znear="0.1" force="0.05"/>
        </visual>

        <statistic extent="4"/>

        <worldbody>
            <light directional="true" diffuse=".8 .8 .8" pos="0 0 10" dir="0 0 -10"/>

            <!-- <geom pos="0 0 0" type="plane" size="3 3 .5" rgba=".7 .7 .7 1" material="MatPlane"/> -->
            
            <!-- <body pos="0 0 0"> -->
            <geom type="box" size="5 5 .01" rgba="0 0 0 .1"/>
            <body name="agent1" pos="{} {} 0">
                <geom class='bot' pos='0 0 .11'/>
                <joint name='slide1x' type='slide' pos='1 0 .11' axis='1 0 0' range='-50 50'/>
                <joint name='slide1y' type='slide' pos='1 0 .11' axis='0 1 0' range='-50 50'/>
            </body>
            <body name="agent2" pos="{} {} 0">
                <geom class='bot' pos='0 0 .11'/>
                <joint name='slide2x' type='slide' pos='2 0 .11' axis='1 0 0' range='-50 50'/>
                <joint name='slide2y' type='slide' pos='2 0 .11' axis='0 1 0' range='-50 50'/>
            </body>
            <body name="agent3" pos="{} {} 0">
                <geom class='bot' pos='0 0 .11'/>
                <joint name='slide3x' type='slide' pos='3 0 .11' axis='1 0 0' range='-50 50'/>
                <joint name='slide3y' type='slide' pos='3 0 .11' axis='0 1 0' range='-50 50'/>
            </body>
            <body name="agent4" pos="{} {} 0">
                <geom class='bot' pos='0 0 .11'/>
                <joint name='slide4x' type='slide' pos='4 0 .11' axis='1 0 0' range='-50 50'/>
                <joint name='slide4y' type='slide' pos='4 0 .11' axis='0 1 0' range='-50 50'/>
            </body>
            <!-- </body> -->
            <!-- <body name='bot1' pos='1 1 .1' quat='0 0 0 0'>
                <geom class='bot'/>
                <joint name='slide1' type='slide' pos='1 1 0' axis='1 0 0' range='-1 1' damping='10' />
            </body> -->

            <geom class="border" fromto="-5 5 0 5 5 0"  />
            <geom class="border" fromto="-5 -5 0 5 -5 0"  />
            <geom class="border" fromto="5 5 0 5 -5 0"  />
            <geom class="border" fromto="-5 5 0 -5 -5 0"  />
            <geom class="borderpost" pos="5 5 0"/>
            <geom class="borderpost" pos="-5 5 0"/>
            <geom class="borderpost" pos="5 -5 0"/>
            <geom class="borderpost" pos="-5 -5 0"/>

            <body name='target' pos='{} {} 1.4' childclass="humanoid">
                <freejoint name="root"/>
                <geom name='torso1' type='capsule' fromto='0 -.07 0 0 .07 0'  size='0.07'/>
                <geom name='head' type='sphere' pos='0 0 .19' size='.09'/>
                <geom name='uwaist' type='capsule' fromto='-.01 -.06 -.12 -.01 .06 -.12' size='0.06'/>
                <body name='lwaist' pos='-.01 0 -0.260' quat='1.000 0 -0.002 0' >
                    <geom name='lwaist' type='capsule' fromto='0 -.06 0 0 .06 0'  size='0.06' />
                    <joint name='abdomen_z' type='hinge' pos='0 0 0.065' axis='0 0 1' range='-45 45' damping='5' stiffness='20' armature='0.02' />
                    <joint name='abdomen_y' type='hinge' pos='0 0 0.065' axis='0 1 0' range='-75 30' damping='5' stiffness='10' armature='0.02' />
                    <body name='pelvis' pos='0 0 -0.165' quat='1.000 0 -0.002 0' >
                        <joint name='abdomen_x' type='hinge' pos='0 0 0.1' axis='1 0 0' range='-35 35' damping='5' stiffness='10' armature='0.02' />
                        <geom name='butt' type='capsule' fromto='-.02 -.07 0 -.02 .07 0'  size='0.09' />
                        <body name='right_thigh' pos='0 -0.1 -0.04' >
                            <joint name='right_hip_x' type='hinge' pos='0 0 0' axis='1 0 0' range='-25 5'   damping='5' stiffness='10' armature='0.01' />
                            <joint name='right_hip_z' type='hinge' pos='0 0 0' axis='0 0 1' range='-60 35'  damping='5' stiffness='10' armature='0.01' />
                            <joint name='right_hip_y' type='hinge' pos='0 0 0' axis='0 1 0' range='-120 20' damping='5' stiffness='20' armature='0.01' />
                            <geom name='right_thigh1' type='capsule' fromto='0 0 0 0 0.01 -.34'  size='0.06' />
                            <body name='right_shin' pos='0 0.01 -0.403' >
                                <joint name='right_knee' type='hinge' pos='0 0 .02' axis='0 -1 0' range='-160 -2' stiffness='1' armature='0.0060' />
                                <geom name='right_shin1' type='capsule' fromto='0 0 0 0 0 -.3'   size='0.049' />
                                <body name='right_foot' pos='0 0 -.39' >
                                    <joint name='right_ankle_y' type='hinge' pos='0 0 0.08' axis='0 1 0'   range='-50 50' stiffness='4' armature='0.0008' />
                                    <joint name='right_ankle_x' type='hinge' pos='0 0 0.04' axis='1 0 0.5' range='-50 50' stiffness='1'  armature='0.0006' />
                                    <geom name='right_foot_cap1' type='capsule' fromto='-.07 -0.02 0 0.14 -0.04 0'  size='0.027' />
                                    <geom name='right_foot_cap2' type='capsule' fromto='-.07 0 0 0.14  0.02 0'  size='0.027' />
                                </body>
                            </body>
                        </body>
                        <body name='left_thigh' pos='0 0.1 -0.04' >
                            <joint name='left_hip_x' type='hinge' pos='0 0 0' axis='-1 0 0' range='-25 5'  damping='5' stiffness='10' armature='0.01' />
                            <joint name='left_hip_z' type='hinge' pos='0 0 0' axis='0 0 -1' range='-60 35' damping='5' stiffness='10' armature='0.01' />
                            <joint name='left_hip_y' type='hinge' pos='0 0 0' axis='0 1 0' range='-120 20' damping='5' stiffness='20' armature='0.01' />
                            <geom name='left_thigh1' type='capsule' fromto='0 0 0 0 -0.01 -.34'  size='0.06' />
                            <body name='left_shin' pos='0 -0.01 -0.403' >
                                <joint name='left_knee' type='hinge' pos='0 0 .02' axis='0 -1 0' range='-160 -2' stiffness='1' armature='0.0060' />
                                <geom name='left_shin1' type='capsule' fromto='0 0 0 0 0 -.3'   size='0.049' />
                                <body name='left_foot' pos='0 0 -.39' >
                                    <joint name='left_ankle_y' type='hinge' pos='0 0 0.08' axis='0 1 0'   range='-50 50'  stiffness='4' armature='0.0008' />
                                    <joint name='left_ankle_x' type='hinge' pos='0 0 0.04' axis='1 0 0.5' range='-50 50'  stiffness='1'  armature='0.0006' />
                                    <geom name='left_foot_cap1' type='capsule' fromto='-.07 0.02 0 0.14 0.04 0'  size='0.027' />
                                    <geom name='left_foot_cap2' type='capsule' fromto='-.07 0 0 0.14  -0.02 0'  size='0.027' />
                                </body>
                            </body>
                        </body>
                    </body>
                </body>
                <body name='right_upper_arm' pos='0 -0.17 0.06' >
                    <joint name='right_shoulder1' type='hinge' pos='0 0 0' axis='2 1 1'  range='-85 60' stiffness='1' armature='0.0068' />
                    <joint name='right_shoulder2' type='hinge' pos='0 0 0' axis='0 -1 1' range='-85 60' stiffness='1'  armature='0.0051' />
                    <geom name='right_uarm1' type='capsule' fromto='0 0 0 .16 -.16 -.16'  size='0.04 0.16' />
                    <body name='right_lower_arm' pos='.18 -.18 -.18' >
                        <joint name='right_elbow' type='hinge' pos='0 0 0' axis='0 -1 1' range='-90 50'  stiffness='0' armature='0.0028' />
                        <geom name='right_larm' type='capsule' fromto='0.01 0.01 0.01 .17 .17 .17'  size='0.031' />
                        <geom name='right_hand' type='sphere' pos='.18 .18 .18'  size='0.04'/>
                    </body>
                </body>
                <body name='left_upper_arm' pos='0 0.17 0.06' >
                    <joint name='left_shoulder1' type='hinge' pos='0 0 0' axis='2 -1 1' range='-60 85' stiffness='1' armature='0.0068' />
                    <joint name='left_shoulder2' type='hinge' pos='0 0 0' axis='0 1 1' range='-60 85'  stiffness='1' armature='0.0051' />
                    <geom name='left_uarm1' type='capsule' fromto='0 0 0 .16 .16 -.16'  size='0.04 0.16' />
                    <body name='left_lower_arm' pos='.18 .18 -.18' >
                        <joint name='left_elbow' type='hinge' pos='0 0 0' axis='0 -1 -1' range='-90 50' stiffness='0' armature='0.0028' />
                        <geom name='left_larm' type='capsule' fromto='0.01 -0.01 0.01 .17 -.17 .17'  size='0.031' />
                        <geom name='left_hand' type='sphere' pos='.18 -.18 .18'  size='0.04'/>
                    </body>
                </body>
            </body>


    <!-- <body name='bot1' pos='1 1 .1' quat='0 0 0 0'>
                <freejoint/>
                <geom class='bot'/>
            </body> -->



            <!-- <body name='bot1' pos='1 1 .1' quat='0 0 0 0'>
                <geom class='bot'/>
                <joint name='slide1' type='slide' pos='1 1 0' axis='1 0 0' range='-1 1' damping='10' />
            </body> -->
            
            <!-- <body pos='1 0 .1' quat='0 0 0 0'>
                <joint name='slide2' type='slide' pos='1 0 0' axis='1 1 0' range='-1 1' damping='10' />
                <geom class='bot'/>
            </body> -->

        </worldbody>

        <actuator>
            <motor name='m1x'    gear='35' joint='slide1x'/>
            <motor name='m1y'    gear='35' joint='slide1y'/>
            <motor name='m2x'    gear='35' joint='slide2x'/>
            <motor name='m2y'    gear='35' joint='slide2y'/>
            <motor name='m3x'    gear='35' joint='slide3x'/>
            <motor name='m3y'    gear='35' joint='slide3y'/>
            <motor name='m4x'    gear='35' joint='slide4x'/>
            <motor name='m4y'    gear='35' joint='slide4y'/>
        </actuator>
    </mujoco>

    """.format(bots[0][0],bots[0][1],bots[1][0],bots[1][1],bots[2][0],bots[2][1],bots[3][0],bots[3][1],x,y)
    return MODEL_XML

MODEL_XML = gen_xml()

class Environment:
    def __init__(self, name='default.xml'):
        #self.model = load_model_from_path("xmls/"+name)
        self.model = load_model_from_xml(MODEL_XML)
        self.sim = MjSim(self.model)
        self.viewer = MjViewer(self.sim)
        self.reset_state = self.sim.get_state()
        self.SPEED = 5
        self.VISION_DIST = 1
        self.destination = [2, 2] #I just randomly chose this
        self.discovered = False
        self.timestep = 0
        self.MAX_TIMESTEP = 200

        #timestep too short for random motion to learn
        #so I step TIME_INC times before next try
        self.TIME_INC = 10

        self.target_previous = np.float32(np.concatenate((self.sim.data.get_body_xpos("target")[:2],
                                self.sim.data.get_body_xvelp("target")[:2]),axis=0))

    #helper function in a class
    @staticmethod
    def dist(x, y): #x and y should be the same length
        return np.linalg.norm(x-y)


    def step(self, actions):
        self.timestep += 1


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

        reward = -.001

        if agent1_dist < self.VISION_DIST or agent2_dist < self.VISION_DIST or agent3_dist < self.VISION_DIST or agent4_dist < self.VISION_DIST:
            if self.discovered == False: # I have this condition so we reward finding the target the first time.
                self.discovered = True
                reward += 10

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

        self.model = load_model_from_xml(gen_xml())
        self.sim = MjSim(self.model)
        # self.viewer = MjViewer(self.sim)
        #self.sim.set_state(self.reset_state)
        self.discovered = False
        self.timestep = 0

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
