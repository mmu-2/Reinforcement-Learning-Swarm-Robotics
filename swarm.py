#!/usr/bin/env python3
"""
Mini-example of Mujoco that demonstrates the key concepts we need.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import os

import random

model = load_model_from_path("xmls/default.xml")
sim = MjSim(model)

viewer = MjViewer(sim)

sim_state = sim.get_state()

#To see order or to change something with the movable parts of
#the agent, check xmls/swarm.xml

#Actions right now:
#sim.data.ctrl[0] = X direction of bot1 [-5 to 5] determines speed
#sim.data.ctrl[1] = Y direction of bot1 [-5 to 5] determines speed
#sim.data.ctrl[2] = X direction of bot2 [-5 to 5] determines speed
#sim.data.ctrl[3] = Y direction of bot2 [-5 to 5] determines speed

while True:
    sim.set_state(sim_state)
    print(sim.data.ctrl)

    # print(sim.data.get_body_xpos("target"))
    # print(sim.data.get_body_xvelp("target"))
    # print(sim.data.get_body_xvelr("target"))

    #sim.data has all the information available in the environment.
    #We should pick and choose some select information.
    # print(dir(sim.data))
    for i in range(1000000): #each iteration and step in this environment is a single frame.

        print('xpos:{}'.format(sim.data.get_body_xpos("target")))
        print('xvelp:{}'.format(sim.data.get_body_xvelp("target")))
        print('xvelr:{}'.format(sim.data.get_body_xvelr("target")))

        #sim.data.ctrl[:] = random.uniform(-1,1)
        sim.data.ctrl[::2] = -4
        #sim.data.ctrl[0] = -1
        #sim.data.ctrl[2] = -1
        # if i % 100 == 0:
        #     print(i)


        sim.step() #step like env.step(action)
        viewer.render() #render like env.render()

    if os.getenv('TESTING') is not None:
        break