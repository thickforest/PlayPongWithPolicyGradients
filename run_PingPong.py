#!/usr/bin/env python
#import dependencies
import gym
import random
import numpy as np
import time
import glob
from RL_brain import PolicyGradient

EPISODES = 999999999
DISPLAY_REWARD_THRESHOLD = -1  # renders environment if total episode reward is greater then this threshold
RENDER = False  # rendering wastes time

#preprocessing function
def prepro(I): #where I is the single frame of the game as the input
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    #the values below have been precomputed through trail and error by OpenAI team members
    I = I[35:195] #cropping the image frame to an extent where it contains on the paddles and ball and area between them
    I = I[::2,::2,0] #downsample by the factor of 2 and take only the R of the RGB channel.Therefore, now 2D frame
    I[I==144] = 0 #erase background type 1
    I[I==109] = 0 #erase background type 2
    I[I!=0] = 1 #everything else(other than paddles and ball) set to 1
    return I.astype('float').ravel() #flattening to 1D


if __name__ == '__main__':
    #implementation details
    env = gym.make('Pong-v0')
    state_size = 6400
    action_size = env.action_space.n

    model_path = ''
    files = glob.glob('RL_model/model-*.meta')
    if files:
        model_path = files[0]

    RL = PolicyGradient(
        n_actions=env.action_space.n,
        #n_features=env.observation_space.shape[0],
        n_features=state_size,
        learning_rate=0.02,
        reward_decay=0.99,
        # output_graph=True,
        save_interval=10,
        load_model=model_path,
    )

    for i_episode in xrange(EPISODES):
        observation = env.reset()
        observation_mod = prepro(observation)

        while True:
            if RENDER: env.render(); time.sleep(0.05)
            action = RL.choose_action(observation_mod)
            next_observation, reward, done, info = env.step(action)
            RL.store_transition(observation_mod, action, reward)

            if done:  #if the episode is over
                ep_rs_sum = sum(RL.ep_rs)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True     # rendering
                print("episode:", i_episode, "  reward:", int(running_reward))
                
                vt = RL.learn()
                break

            observation_mod = prepro(next_observation)
