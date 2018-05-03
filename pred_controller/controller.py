#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 18:33:16 2018
@author: wangweilong
"""

import numpy as np
import pickle
from sklearn import datasets, svm

filename = './rf_model.sav'
load_model = pickle.load(open(filename,'rb'))
gamma = 0.9
# example
# features: temperature_normal, humidity_normal, skin_normal, clothing
test = np.array([[0.784,20,61,27.55,81.55,1.00]])
pred = load_model.predict(test)
# state space of temperature 
states = np.arange(18,25,0.3)
nS = len(states)
# increase or decrease temperature
Action = [-0.01,0,0.01]
nA = len(Action)

def value_iteration(test, gamma, tol=1e-3):
    v = []
    policy = []
    while True:
        delta = float(0)
        value_current = list(v)
        print(value_current)
        value = np.zeros(nA)
        for act in range(nA):
            inputs = test
            inputs[0] = inputs[0]+Action[act]
            pred = load_model.predict(inputs)
            if pred == -3:
                value[act] = value[act] - 5
            elif pred == -2:
                value[act] = value[act] - 2
            elif pred == -1:
                value[act] = value[act] - 1
            elif pred == 0:
                value[act] = value[act] + 1
            elif pred == 1:
                value[act] = value[act] - 1
            elif pred == 2:
                value[act] = value[act] - 2
            elif pred == 3:
                value[act] = value[act] - 5 
            max_value = max(value)
            policy = max([act for act, vl in enumerate(value) if vl == max_value])
            print(policy)
            print(max_value)
            delta = max(delta, abs(max_value-value_current))
            v = max_value
        if delta < tol:
            break
    return policy, v

policy, v = value_iteration(test, gamma, tol=1e-3)