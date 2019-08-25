#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 19:48:10 2019

@author: Asher
"""

# import numpy as np


class ActivationFunction:

    """
    X of shape NxM where each of the M column vectors is a point of dim Nx1.
    W of shape KxN, where K is the number of possible classes.
    b of shape Kx1.
    y of shape KxN where each column is the one-hot vector for one of the N points.
    
    k_probs  of shape KxM. For each class, stores the the probability that each of the M points belong to that class.
    loss - the log loss. 
    """
    
    def __init__(self,N, K, W, b, W2=None):
        self.N = N
        self.K = K
        self.W = W
        self.b = b
        self.W2 = W2
        self.X = None
        self.dW = None
        self.db = None
        self.dX = None

        

    # def update_weights(self, W, b):
    #     self.W = W
    #     self.b = b

    def forward(self, X, y=None):
        raise NotImplementedError
        
    
    def backward(self, input_grad):
        raise NotImplementedError
