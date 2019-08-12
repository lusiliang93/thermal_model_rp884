#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 18:33:16 2018
@author: siliang lu
check the correctness!! @12.28
"""

import numpy as np
import pickle
from sklearn import datasets, svm
import random

def reward_calc(state):
    '''
    filename = 'ensemble_model.sav'
    load_model = pickle.load(open(filename,'rb'))
    pred = load_model.predict(state.reshape(1,-1))[0]
    print(pred)
    pred = load_model.predict(state.reshape(1,-1))[0]
    if pred == 3:
        reward = 10000
        is_done = True
    elif pred == 0:
        reward = -10
        is_done = False
    elif pred == 1:
        reward = -100
        is_done = False
    elif pred == 2:
        reward = -1000
        is_done = False
    elif pred == 4:
        reward = -10
        is_done = False
    elif pred == 5:
        reward = -100
        is_done = False
    elif pred == 6:
        reward = -1000
        is_done = False
    '''
    # penalize the lower bound
    # take clo and rh into account
    if state[1] < 16:
        reward = -1000
        is_done = False
    elif state[1] < 23 and state[1] > 15:
        reward = -1*(23-state[1])
        is_done = False
    elif state[1] > 22 and state[1] < 26:
        reward = 10
        is_done = True
    elif state[1] > 25 and state[1] < 32:
        reward = -1*(state[1]-25)
        is_done = False
    # penalize the upper bound
    else:
        reward = -1000
        is_done = False
    return reward, is_done

def next(act,state,states):
    old_index = states.tolist().index(state.tolist())
    if act == 0:
        return state, old_index
    elif act == -1:
        index = old_index-1
        nexts = states[index]
    else:
        index = old_index+1
        nexts = states[index]
    return nexts, index

def act(Actions, s, Q, pos, states):
    q = []
    if s[1] == 15:
        return 1, 2
    elif s[1] == 31:
        return -1, 0
    else:
        for index, act in enumerate(Actions):
            nexts,_ = next(act,s,states)
            reward, is_done = reward_calc(nexts)
            qa = Q[pos][index] + reward
            q.append(qa)
        index = np.argmax(np.array(q))
        a = Actions[index]
        return a, index

def train(states, Actions):
    gamma = 0.9 
    alpha = 0.1
    epsilon = 0.1

    # initialize Q-table
    Q = np.zeros((len(temps),len(Actions)))
    # change # of episodes
    for i in range(0,100):
        init = random.choice([1,15])
        s = states[init]
        is_done = False

        while not is_done:
            if random.uniform(0,1) < epsilon:
                ind= random.randint(0,2)
                action = Actions[ind]
                if (s[1] == 15 and action == -1) or (s[1] == 31 and action == 1):
                    continue
            else:
                action, ind = act(Actions,s,Q,init,states)

            # next state and its index
            nexts, index = next(action,s,states)
            reward, is_done = reward_calc(nexts)
            old_value = Q[init][ind]
            # max of Q-value of next state
            next_max = np.max(Q[index])
            new_value = (1-alpha) * old_value + alpha *(reward+gamma*next_max)
            Q[init][ind] = new_value
            s = nexts
            init = states.tolist().index(s.tolist())
    return Q

def evaluate(state, states, Actions, Q):
    index = states.tolist().index(state.tolist())
    # force to remain in the boundary
    if index == 0:
        a = 1
    # force to remain in the boundary
    elif index == 16:
        a = -1
    else:
        a = Actions[np.argmax(Q[index])]
    return a

if __name__=='__main__':
    # features: clo, temperature, relative humidity
    # the interval must corrspond to the action
    # the upper and lower bounds must be parsed into reward func
    temps = np.arange(15,32,1)
    states=np.zeros((len(temps),3))
    states[:,0] = 0.5
    states[:,2] = 60
    states[:,1] = temps
    # increase or decrease temperature
    Actions = [0,1,-1]
    Q=train(states, Actions)
    print(Q)
    episode = []
    ''' generate policies with different start state
    for i in range(10):
    	#state = np.array([0.5,31,60])
    	state = [0]* 3
    	# clo (0.5,1)
    	state[0] = 0.5
    	# humidity (20, 60, 80)
    	state[2] = 60
    	# temperature(18, 25, 30)
    	state[1] = 18
    	# random initialize action
    	a = -2
    	# the drawback is that humidity and clo should be constant
    	policy = []
    	t = 0 #timestamp
    	while (a is not 0):
    		a = evaluate(np.array(state), states, Actions, Q)
    		state[1] += a
    		policy.append(a)
    		t += 1
    		if (t>100):
    			break
    	episode.append(policy)
    '''
