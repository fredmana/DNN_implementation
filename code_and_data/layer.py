#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 19:48:10 2019

@author: Asher
"""

import numpy as np
from softmax import Softmax
from tanh import Tanh
from relu import Relu
from sigmoid import Sigmoid

class Layer:

    """
    This class represents a layer in the deep network. Each layer has a specific id based on its location in the net.
    Each layer can call one or more activation functions (e.g. for resnet will call 'linear activation + relu').
    A layer holds a reference to a global dictionary which holds the the
    values generated during the forward and backward passes.
    The layer knows what activation functions it needs to run (which are stored in a list of strings)
    and whether or not it needs to also compute the loss of the network (i.e. whether it's the last layer in training).
    """
    
    def __init__(self, id, N, K, activation_names, compute_loss, layers_dict, is_resnet=False):
        self.id = id
        self.N = N
        self.K = K
        self.activation_dict = {"softmax": Softmax, "tanh": Tanh, "relu": Relu, "sigmoid": Sigmoid}
        self.activation_name = activation_names[0] #for now there will only be one activation func.
                                                   #later for resnet will need to iterate of the list.
        self.layer_info = {}
        # if self.activation_name == "sigmoid":
        #     self.K = K-1 # for sigmoid we want a output of one row with shape (1,M)
        # np.random.seed(5)
        self.is_resnet = is_resnet
        self.layer_info['W'] = np.random.randn(self.K, self.N) * 0.1
        self.layer_info['b'] = np.random.randn(self.K,1) * 0.1
        if self.is_resnet:
            self.layer_info['W2'] = np.random.randn(self.N, self.K) * 0.1  # TODO
            self.activation_func = self.activation_dict[self.activation_name]\
                (self.N, self.K, self.layer_info['W'], self.layer_info['b'], self.layer_info['W2'])
        else:
            self.activation_func = self.activation_dict[self.activation_name] \
                (self.N, self.K, self.layer_info['W'], self.layer_info['b'])
        # self.layer_info['W'] = np.concatenate((W,b), axis=1)
        self.compute_loss = compute_loss
        self.layers_dict = layers_dict

        
    def forward(self, X, y=None):
        M = X.shape[1]
        # X_combined = np.concatenate((X, np.ones((1, M))), axis=0)
        forward_out = self.activation_func.forward(X, y)
        self.layer_info['activation_output'] = forward_out['activation_output']
        if self.compute_loss:
            self.layer_info['loss'] = forward_out['loss']
            self.layer_info['accuracy'] = forward_out['accuracy']



    def backward(self, lr, input_grad=None, gradient_testing=False):
        if self.is_resnet:
            dW, db, dX, dW2 = self.activation_func.backward(input_grad) #since I dont really use b, db will be None
            self.layer_info['dW2'] = dW2
        else: #for example in the softmax case where w2 isn't relevant
            dW, db, dX = self.activation_func.backward(input_grad) #since I dont really use b, db will be None
        # print("%%%%%%% dW shape is {}".format(dW.shape))
        # print("%%%%%%% W shape is {}".format(self.layer_info['W'].shape))
        self.layer_info['dW'] = dW
        self.layer_info['db'] = db
        self.layer_info['dX'] = dX
        # average_dW = np.mean(self.layer_info['dW'], axis=0)
        # self.layer_info['W'] -= lr * average_dW[:,np.newaxis].T
        if gradient_testing==False:
            self.layer_info['W'] -= lr * self.layer_info['dW']
            self.layer_info['b'] -= lr * self.layer_info['db']
            if self.is_resnet:
                self.layer_info['W2'] -= lr * self.layer_info['dW2']  # TODO
            # self.activation_func.update_weights(self.layer_info['W'], self.layer_info['b'])



    def set_w(self, W):  # for testing purposes.
        self.layer_info['W'] = W
        self.activation_func.set_w(W)


    def set_b(self, b):  # for testing purposes.
        self.layer_info['b'] = b
        self.activation_func.set_b(b)


    def set_w2(self, W2):  # for testing purposes.
        self.layer_info['W2'] = W2
        self.activation_func.set_w2(W2)

    def set_params(self, W, b, W2=None):
        self.set_w(W)
        self.set_b(b)
        if self.is_resnet:
            self.set_w2(W2)