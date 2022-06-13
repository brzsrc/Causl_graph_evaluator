#!/usr/bin/env python

#SCM imports
import structeral_causal_modeling as scm


#cartPole import
import gym
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
torch.manual_seed(0)

import sys
sys.path.append("../..")
from helpers.config_simulations import get_config

config, _ = get_config()
train_causal = config.train_scm

"""Environment Init and Config"""
config.train = not config.eval

#---------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ### Instantiate the Environment and Agent
# 
# CartPole environment is very simple. It has discrete action space (2) and 4 dimensional state space. 
env = gym.make('CartPole-v1')
env.reset(seed=0)

print('observation space:', env.observation_space)
print('action space:', env.action_space)

# ### Define Policy
# Unlike value-based method, the output of policy-based method is the probability of each action. It can be represented as policy. So activation function of output layer will be softmax, not ReLU.
class Policy(nn.Module):
    def __init__(self, state_size=4, action_size=2, hidden_size=32):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        # we just consider 1 dimensional probability of action
        return F.softmax(x, dim=1)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        model = Categorical(probs)
        action = model.sample()
        return action.item(), model.log_prob(action)


# ### REINFORCE
def reinforce(policy, optimizer, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    scores_deque = deque(maxlen=100)
    scores = []
    replay_obs = []
    replay_act = []
    for e in range(1, n_episodes):
        saved_log_probs = []
        rewards = []
        state = env.reset()

        obs_set = []
        action_set = [] 

        # Collect trajectory
        for step in range(max_t):
            step+=1
            # Sample the action from current policy
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)

            action_set.append(action)
            obs_set.append(state.tolist() + [reward])


            if done:
                break

        # Calculate total expected reward
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        
        # Recalculate the total reward applying discounted factor
        discounts = [gamma ** i for i in range(len(rewards) + 1)]
        R = sum([a * b for a,b in zip(discounts, rewards)])
        
        # Calculate the loss 
        policy_loss = []
        for log_prob in saved_log_probs:
            # Note that we are using Gradient Ascent, not Descent. So we need to calculate it with negative rewards.
            policy_loss.append(-log_prob * R)
        # After that, we concatenate whole policy loss in 0th dimension
        policy_loss = torch.cat(policy_loss).sum()
        
        # Backpropagation
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        # if train_causal:
        if 1 :
            if len(replay_obs) < 1:
                replay_obs = obs_set
                replay_act = action_set
            else:
                replay_obs.extend(obs_set)
                replay_act.extend(action_set)

        act_array = np.array(action_set)
        obs_array = np.array(obs_set)
        # print("act shape:", act_array.shape)
        # print("obs shape:", obs_array.shape)

        # act_array = np.array(replay_act)
        # obs_array = np.array(replay_obs)
        # print("replay_act shape:", act_array.shape)
        # print("replay_obs shape:", obs_array.shape)
                                
        if len(replay_obs) >= config.data_size:

            """generate why and why not explanations for a given state index of the batch data (here 0) and save to file"""
            scm.process_explanations(replay_obs, replay_act, config, 0, step)    
            replay_obs = []
            replay_act = []


        if e % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(e, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(e - 100, np.mean(scores_deque)))
            break
    return scores

#Run the training process
def training_agent(policy):
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    scores = reinforce(policy, optimizer, n_episodes=2000)


policy = Policy().to(device)
training_agent(policy)




