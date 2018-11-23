#!/usr/bin/env python
#coding:utf-8
#import dependencies
import numpy as np   #for matrix math
import cPickle as pickle  #to save/load model
import gym
import time


#initialise : init model
D = 80*80 #input dimension
model = pickle.load(open('model.v','rb'))


#activation function
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))   #adding non linearing + squashing

def relu(x):
    x[x<0] = 0
    return x


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


def policy_forward(x):
    h = np.dot(model['W1'],x)   
    h = relu(h)  
    logit = np.dot(model['W2'],h)
    p = sigmoid(logit)
    return p,h   #probability of action 2(i.e. UP) and hidden layer state i.e. hidden state


#implementation details
env = gym.make('Pong-v0')
observation = env.reset()
prev_x = None #prev frame value in order to compute the difference between current and previous frame
#as discussed frames are static and the difference is used to capture the motion
#Intially None because there's no previous frame if the current frame is the 1st frame of the game
running_reward = None
reward_sum = 0
episode_number = 0


#begin training
while True:
    env.render()
    time.sleep(0.1)
    #get the input and preprocess it
    cur_x = prepro(observation)
    #get the frame difference which would be the input to the network
    if prev_x is None:
        prev_x = np.zeros(D)
    x = cur_x - prev_x
    prev_x = cur_x
    
    #forward propagation of the policy network
    #sample an action from the returned probability
    aprob, h = policy_forward(x)
    #stochastic part
    if 0.5 < aprob:	# 均匀分布
        action = 2
    else:
        action = 3
    
    if action == 2:
        y = 1
    else:
        y = 0
    
    #new step in the environment
    observation, reward, done, info = env.step(action)
    reward_sum += reward #for advantage purpose
    
    if done:  #if the episode is over
        episode_number+=1
        
        print('Episode Reward : {}'.format(reward_sum))
        
        reward_sum = 0
        prev_x = None
        observation = env.reset() #resetting the environment since episode has ended
    # end if done:  #if the episode is over
