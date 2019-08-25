#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 19:48:10 2019

@author: Asher
"""

import numpy as np
from activation_func import ActivationFunction


class Softmax(ActivationFunction):
    """
    X of shape NxM where each of the M column vectors is a point of dim Nx1.
    W of shape KxN, where K is the number of possible classes.
    b of shape Kx1.
    y of shape KxN where each column is the one-hot vector for one of the N points.
    
    k_probs  of shape KxM. For each class, stores the the probability that each of the M points belong to that class.
    loss - the log loss. 
    """

    def __init__(self, N, K, W, b):
        ActivationFunction.__init__(self, N, K, W, b)
        self.M = None
        self.probs = None
        self.loss = 0

    def forward(self, X, y=None):
        self.X = X

        self.y = y
        self.M = X.shape[1]

        forward_output = {}

        # f = np.dot(self.X.T, self.W.T) + self.b.T #ch
        f = np.dot(self.W, self.X) + self.b
        f -= np.matrix(np.max(f, axis=0))

        sum_j = np.sum(np.exp(f), axis=0)
        # self.probs = np.exp(f) / np.matrix(sum_j).T #ch
        self.probs = np.exp(f).T / np.matrix(sum_j).T

        self.probs = np.array(self.probs)
        log_prob = -np.log(self.probs)

        # In this case we are'nt in the training/testing phase,
        # and the already trained network is being used for real data,
        # so just return for each point the class with the highest probability.
        # After taking the argmax we need to add one since the classes are one-based.
        if self.y is None:
            return None, (np.argmax(log_prob, axis=0) + 1)
        else:
            self.y = self.y.astype(int)

        log_loss = np.multiply(self.y, log_prob)
        self.loss = np.sum(log_loss) / self.M

        self.accuracy = self.compute_accuracy(self.probs, self.y)

        forward_output['activation_output'] = self.probs
        forward_output['loss'] = self.loss
        forward_output['accuracy'] = self.accuracy

        return forward_output

    def backward(self, input_grad=None):
        # self.probs = self.probs
        self.y = self.y.reshape(self.probs.shape)
        input_grad = (self.probs - self.y).T
        self.dW = self.w_grad(input_grad)
        self.db = self.b_grad(input_grad)
        self.dX = self.x_grad(input_grad)
        return self.dW, self.db, self.dX

    def w_grad(self, input_grad=None):
        if input_grad is None:  # if were testing the gradient
            self.y = self.y.reshape(self.probs.shape)
            input_grad = (self.probs - self.y).T

        self.dW = np.dot(input_grad, self.X.T) / self.M
        return self.dW

    def b_grad(self, input_grad):
        self.db = np.sum(input_grad, axis=1, keepdims=True) / self.M
        return self.db

    def x_grad(self, input_grad):
        self.dX = np.dot(self.W.T, input_grad)
        return self.dX

    def set_w(self, W):  # for testing purposes.
        self.W = W

    def set_b(self, b):  # for testing purposes.
        self.b = b

    def compute_accuracy(self, probs, y):
        true_labels = np.argmax(y, 1)
        y_hat = np.argmax(probs, 1)
        return np.mean(true_labels == y_hat)
