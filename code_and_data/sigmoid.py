#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 19:48:10 2019

@author: Asher
"""

import numpy as np
import scipy.io
from utils import *
from activation_func import ActivationFunction

class Sigmoid(ActivationFunction):
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

        Z = np.dot(self.W, self.X) + self.b

        self.probs = 1 / (1 + np.exp(-Z))
        binary_prob_array = np.concatenate((self.probs, 1-self.probs),0).T

        log_prob = -np.log(self.probs)

        log_loss = np.multiply(self.y, log_prob)
        self.loss = np.sum(log_loss)/self.M

        self.accuracy = self.compute_accuracy(binary_prob_array, y)

        forward_output['activation_output'] = self.probs
        forward_output['loss'] = self.loss
        forward_output['accuracy'] = self.accuracy

        return forward_output




    def backward(self, input_grad=None):
        true_labels = np.argmax(self.y, 0)
        # true_labels = true_labels.reshape(self.probs.shape)
        # sigmoid_grad = - (np.divide(true_labels, self.probs) - np.divide(1 - true_labels, 1 - self.probs))
        sigmoid_grad = self.probs * (1 - self.probs)
        self.dW = self.w_grad(sigmoid_grad)
        self.db = self.b_grad(sigmoid_grad)
        self.dX = self.x_grad(sigmoid_grad)
        return self.dW, self.db, self.dX
    
    
    def w_grad(self, sigmoid_grad):
        self.dW = np.dot(sigmoid_grad, self.X.T) / self.M
        return self.dW

    def b_grad(self, sigmoid_grad):
        self.db = np.sum(sigmoid_grad, axis=1, keepdims=True) / self.M
        return self.db


    def x_grad(self, sigmoid_grad):
        self.dX = np.dot(self.W.T, sigmoid_grad)
        return self.dX

    
    def set_w(self, W): #for testing purposes. 
        self.W = W


    def compute_accuracy(self, probs, y):
        true_labels = np.argmax(y, 0)
        y_hat = np.argmax(probs, 1)
        return np.mean(true_labels == y_hat)

    # an auxiliary function that converts probability into class
    def convert_prob_into_class(probs):
        probs_ = np.copy(probs)
        probs_[probs_ > 0.5] = 1
        probs_[probs_ <= 0.5] = 0
        return probs_


def main():  
    
    np.random.seed(42)
    
    # data_path = '/Users/Asher/Desktop/my_Documents/MSc/courses/optimization/Project_2019/data/SwissRollData.mat'
    data_path = '/Users/Asher/Desktop/my_Documents/MSc/courses/optimization/Project_2019/data/PeaksData.mat'
    # data_path = '/Users/Asher/Desktop/my_Documents/MSc/courses/optimization/Project_2019/data/GMMData.mat'
    data = scipy.io.loadmat(data_path)
    Xt = data["Yt"]
    Yt = data["Ct"]
    Xv = data["Yv"]
    Yv = data["Cv"]
    print("shapes are X_train:{}, y_train:{}, X_val:{}, y_val:{}"
          .format(Xt.shape,Yt.shape,Xv.shape,Yv.shape))
    X = Xv[:,1][:,np.newaxis]
    y = Yv[:,1][:,np.newaxis]
    
    image_size, num_images = X.shape
    num_labels = y.shape[0]

    W = np.random.randn(num_labels, image_size)
    b = np.zeros((num_labels, 1))

    W_combined = np.concatenate((W, b), axis=1)
    X_combined = np.concatenate((X, np.ones((1,num_images))), axis=0)
    
    # input_shape = (image_size, num_images)
    # output_shape = (num_labels, num_images)
    
    classifier = Sigmoid(image_size+1, num_labels, W_combined)
    
    gradient_test(classifier, X_combined, y, (num_labels, image_size+1))

       
if __name__ == "__main__":
    main()
    