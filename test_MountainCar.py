#!/usr/bin/env python
#import dependencies
import gym
import random
import numpy as np
import time
import glob
from RL_brain import PolicyGradient

TIME_INTERVAL=0.05

if __name__ == '__main__':
    #implementation details
    env = gym.make('MountainCar-v0')
    env.seed(1)
    env = env.unwrapped
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    RL = PolicyGradient(
        n_actions=env.action_space.n,
        #n_features=env.observation_space.shape[0],
        n_features=state_size,
        learning_rate=1e-4,
        reward_decay=0.99,
        # output_graph=True,
        # save_interval=10,
        resume=True,
        work_dir="MountainCarModel",
    )

    i_episode = 0
    while True:
        i_episode += 1
        observation = env.reset()

        #print observation
        step = 0
        while True:
            step += 1
            env.render()
            time.sleep(TIME_INTERVAL)
            #action = RL.max_choose_action(observation)
            action = RL.random_choose_action(observation)
            #print "ACTION:", action
            next_observation, reward, done, info = env.step(action)
            #print next_observation, reward, done
            #RL.store_transition(observation_mod, action, reward)
            if done:  #if the episode is over
                print("episode:", i_episode, "  score:", -step)
                
                #vt = RL.learn()
                break

            observation = next_observation
