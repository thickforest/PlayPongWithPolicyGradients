#!/usr/bin/python
"""
Policy Gradient, Reinforcement Learning.

The cart pole example

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt
import time
import sys

env = gym.make('MountainCar-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

#print(env.action_space)
#print(env.observation_space)
#print(env.observation_space.high)
#print(env.observation_space.low)

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=1e-4,
    reward_decay=0.99,
    # output_graph=True,
    save_interval=10,
    resume=True,
    work_dir="MountainCarModel",
)

i_episode = 0
while True:
    i_episode += 1
    observation = env.reset()

    step = 0
    while True:
        step += 1

        action = RL.random_choose_action(observation)

        observation_, reward, done, info = env.step(action)

        if done: reward = 1000

        RL.store_transition(observation, action, reward)

        if done:
            print("episode:", i_episode, "  score:", -step)

            vt = RL.learn()
            break

        observation = observation_
