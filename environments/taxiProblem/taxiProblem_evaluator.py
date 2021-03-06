#!/usr/bin/env python


#SCM imports
import structeral_causal_modeling as scm


#taxiProblem import
import numpy as np
import gym
import random

import sys
sys.path.append("../..")
from helpers.config_simulations import get_config

config, _ = get_config()
train_causal = config.train_scm

"""Environment Init and Config"""
config.train = not config.eval

#---------------------------------------------------------------
# n = 5, 25 taxi positions
def encode_location(taxi_x, taxi_y, n):
    locat_arr = np.arange(1, n * n + 1).reshape(n,n)
    return locat_arr[taxi_x, taxi_y]



env = gym.make("Taxi-v3")

action_size = env.action_space.n
print("Action size ", action_size)
state_size = env.observation_space.n
print("State size ", state_size)
qtable = np.zeros((state_size, action_size))

total_episodes = 2000        # Total episodes
max_steps = 99                # Max steps per episode
learning_rate = 0.7           # Learning rate
gamma = 0.618                 # Discounting rate

epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability 
decay_rate = 0.01             # Exponential decay rate for exploration prob

"""End of Envronment Initialization"""

"""Agent training"""
replay_obs = []
replay_act = []
for episode in range(total_episodes):
    # print("episode: ", episode)
    # Reset the environment
    state = env.reset()
    step = 0
    done = False
    obs_set = []
    action_set = []
    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        step += 1
        exp_exp_tradeoff = random.uniform(0,1)
        
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state,:])
        
        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()
        
        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)
        new_taxi_row, new_taxi_col, new_pass_idx, new_dest_idx = env.decode(new_state)

        action_ = action
        if action != 4 and action != 5:
            action_ = 6
        action_set.append(action_)
        taxi_loc = encode_location(new_taxi_row, new_taxi_col, 5)
        obs_set.append([taxi_loc, new_pass_idx, new_dest_idx, reward])

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * 
                                    np.max(qtable[new_state, :]) - qtable[state, action])
                
        # Our new state is state
        state = new_state
        
        # If done : finish episode
        if done == True: 
            break
    
    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 

    if len(replay_obs) < 1:
        replay_obs = obs_set
        replay_act = action_set
    else:
        replay_obs.extend(obs_set)
        replay_act.extend(action_set)
                            
    if len(replay_obs) >= config.data_size:

        """generate why and why not explanations for a given state index of the batch data (here 0) and save to file"""
        scm.process_explanations(replay_obs, replay_act, config, 0, step)    
        replay_obs = []
        replay_act = []


env.close()


