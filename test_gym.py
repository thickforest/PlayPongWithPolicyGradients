#!/usr/bin/env python
#import dependencies
import gym
import random
import time
import numpy as np


if __name__ == '__main__':
    #implementation details
    env = gym.make('Pong-v0')

    observation = env.reset()

    #begin training
    while True:
        env.render()
        #get the input and preprocess it
        action = random.randint(2,3)
        if action == 2:
            print('UP')
        else:
            print('DOWN')
        print(action)
        observation, reward, done, _ = env.step(action)
        #print(observation, reward, done)
        print(observation.shape, reward, done)
        time.sleep(0.1)
        
        if done:  #if the episode is over
            observation = env.reset() #resetting the environment since episode has ended
            time.sleep(5)
        # end if done:  #if the episode is over
