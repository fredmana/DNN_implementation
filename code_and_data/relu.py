#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 19:48:10 2019

@author: Asher
"""

import numpy as np
from activation_func import ActivationFunction


class Relu(ActivationFunction):
    """
    X of shape NxM where each of the M column vectors is a point of dim Nx1.
    W of shape KxN, where K is the number of possible classes.
    b of shape Kx1.

    """

    def __init__(self, N, K, W, b):
        ActivationFunction.__init__(self, N, K, W, b)
        self.M = None
        self.output = None
        self.relu_grad = None

    def forward(self, X, y=None):
        self.X = X
        self.M = X.shape[1]

        forward_output = {}

        self.Z = np.dot(self.W, self.X) + self.b
        self.output = np.maximum(0, self.Z)

        forward_output['activation_output'] = self.output
        return forward_output

    def backward(self, input_grad):
        relu_grad = np.array(input_grad, copy=True)
        relu_grad[self.Z <= 0] = 0
        self.relu_grad = relu_grad

        self.dW = self.w_grad(self.relu_grad)
        self.db = self.b_grad(self.relu_grad)
        self.dX = self.x_grad(self.relu_grad)
        return self.dW, self.db, self.dX

    def w_grad(self, relu_grad):
        self.dW = np.dot(relu_grad, self.X.T) / self.M
        return self.dW

    def b_grad(self, relu_grad):
        self.db = np.sum(relu_grad, axis=1, keepdims=True) / self.M
        return self.db

    def x_grad(self, relu_grad):
        self.dX = np.dot(self.W.T, relu_grad)
        # self.dX = np.sum(self.dX, axis=0, keepdims=True) / self.M
        return self.dX
