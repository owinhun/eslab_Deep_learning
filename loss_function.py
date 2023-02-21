#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np

def one_hot(y, length_of_onehot):
    return np.eye(length_of_onehot)[y].reshape(-1, length_of_onehot)

def crossentropy(z, y,epsilon=0.0001):  # sample의 평균 // 클래스간의 합 z = pred, y = label
    return np.mean(-np.sum((one_hot(y, 10) * np.log(z+epsilon)), axis = 1))

def ave_accuracy(output, y):
    # print(np.argmax(output, axis = 1))
    # print(np.argmax(output, axis = 1).reshape(20, 1))
    return np.mean(np.argmax(output, axis=1).reshape(-1, 1) == y, axis=0)