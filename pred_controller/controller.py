#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 18:33:16 2018

@author: wangweilong
"""

import numpy as np
import pickle
from sklearn import datasets, svm


filename = 'svm_model.sav'
load_model = pickle.load(open(filename,'rb'))
gamma = 0.9
# example
# features: temperature_normal, humidity_normal, skin_normal, clothing
test = np.array([[-1.785714,-0.457604,-2.04924,21.7]])
pred = load_model.predict(test)
states = np.arange(18,25,0.5)
nS = len(states)
Action = [-0.5,0,0.5]
nA = len(Action)

def value_iteration(test, gamma, tol=1e-3):
    v = np.zeros(nS)
    policy = np.zeros(nS)
    while True:
        delta = float(0)
        value_current = list(v)
        for i in range(nS):
            value = np.zeros(nA)
            for act in range(nA):
                inputs = test
                inputs[0] = states[i]
                pred = load_model.predict(inputs)
                if pred == -3:
                    value[act] = value[act] - 10
                elif pred == -2:
                    value[act] = value[act] - 5
                elif pred == -1:
                    value[act] = value[act] - 1
                elif pred == 0:
                    value[act] = value[act] + 10
                elif pred == 1:
                    value[act] = value[act] - 1
                elif pred == 2:
                    value[act] = value[act] - 5
                elif pred == 3:
                    value[act] = value[act] - 10  
            max_value = max(value)
            policy[i] = max([act for act, vl in enumerate(value) if vl == max_value])
            delta = max(delta, abs(max_value-value_current[i]))
            v[i] = max_value
        if delta < tol:
            break
    return policy

policy = value_iteration(test, gamma, tol=1e-3)
print(policy)