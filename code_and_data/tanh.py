#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 19:48:10 2019

@author: Asher
"""

import numpy as np
from activation_func import ActivationFunction

class Tanh(ActivationFunction):
    """
    X of shape NxM where each of the M column vectors is a point of dim Nx1.
    W of shape KxN, where K is the number of possible classes.
    b of shape Kx1.
    y of shape KxN where each column is the one-hot vector for one of the N points.

    k_probs  of shape KxM. For each class, stores the the probability that each of the M points belong to that class.
    loss - the log loss.
    """

    def __init__(self, N, K, W, b, W2=None):
        ActivationFunction.__init__(self, N, K, W, b, W2)
        self.M = None
        self.output = None
        self.tanh_grad = None



    def forward(self, X, y=None):
        self.X = X
        self.M = X.shape[1]

        forward_output = {}

        self.Z1 = np.dot(self.W, self.X) + self.b

        self.tanh_output = np.tanh(self.Z1)
        self.output = self.tanh_output
        if self.W2 is not None: #if this should be a resnet layer
            self.Z1 = np.dot(self.W, self.X) + self.b
            self.output = np.dot(self.W2.T, self.X) + self.tanh_output

        forward_output['activation_output'] = self.output

        return forward_output



    def backward(self, input_grad=None):
        self.input_grad = input_grad
        self.tanh_grad = 1 - np.square(self.tanh_output)
        self.chained_grad = np.multiply(self.tanh_grad, input_grad)

        self.dW = self.w_grad(self.input_grad, self.tanh_grad, self.chained_grad)
        self.db = self.b_grad(self.input_grad, self.tanh_grad, self.chained_grad)
        self.dX = self.x_grad(self.input_grad, self.tanh_grad, self.chained_grad)
        if self.W2 is not None:  # if this is a resnet layer
            self.dW2 = self.w2_grad(self.tanh_output, self.input_grad)
            return self.dW, self.db, self.dX, self.dW2
        return self.dW, self.db, self.dX



    def b_grad(self, input_grad, tanh_grad, chained_grad):
        self.db = np.sum(chained_grad, axis=1, keepdims=True) / self.M
        return self.db


    def w_grad(self, input_grad, tanh_grad, chained_grad):
        self.dW = np.dot(chained_grad, self.X.T) / self.M
        return self.dW

    def x_grad(self, input_grad, tanh_grad, chained_grad):
        if self.W2 is not None:  # if this is a resnet layer
            right_side = np.dot(self.W.T, chained_grad)
            left_side = np.dot(self.W2, input_grad)
            self.dX = left_side + right_side
        else:
            self.dX = np.dot(self.W.T, chained_grad)
        return self.dX

    def w2_grad(self, tanh_output, input_grad):
        if self.W2 is not None:  # if this is a resnet layer
            self.dW2 = np.dot(input_grad, self.X.T).T / self.M
        else:
            self.dW2 = None
        return self.dW2

    def set_w(self, W):  # for gradient testing purposes.
        self.W = W

    def set_b(self, b):  # for testing purposes.
        self.b = b

    def set_w2(self, W2):  # for gradient testing purposes.
            self.W2 = W2

