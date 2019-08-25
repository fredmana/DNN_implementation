#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 19:48:10 2019

@author: Asher
"""

import matplotlib.pyplot as plt
import scipy.io
from neural_network import *
from tanh import *


def gradient_test(test_type, is_res, data_path, iterations=10, num_hidden_layers=None, hidden_layers_size=None):
    '''
    params:
     - test_type: indicates the type of gradient test needed to be peformed.
        options:
        1. softmax. Here will will test the grad of the param W of the function softmax(Wx + b)
        2. tanh. Here will will test the grad of the param W1 of the function softmax(W2(tanh(W1x + b1)) + b2)
        3. whole_network. Here we will run gradient tests for all W's and b's in the network
            which will generally be of length >= 3.
     - is_res: should we check the gardient of a residual activation function.
     - data_name: the name of the data file.
     - iterations: number of iterations for the gradient check.
     - in the case of a deep neural network testing we have two extra params:
        1. num_hidden_layers
        2. hidden_layers_size (number of neurons)
    '''
    data, grad_tester, shapes = prepare_tester_and_data(test_type, is_res, data_path, num_hidden_layers, hidden_layers_size)
    run_grad_tester(data, grad_tester, shapes, test_type, is_res, iterations)


def prepare_tester_and_data(test_type, is_res, data_path, num_hidden_layers=None, hidden_layers_size=None):
    np.random.seed(37)
    all_data = scipy.io.loadmat(data_path)
    data = {}

    data['X_train'] = all_data["Yt"][:,1][:,np.newaxis]
    data['y_train'] = all_data["Ct"][:,1][:,np.newaxis].T
    data['X_val'] = all_data["Yv"][:,1][:,np.newaxis]
    data['y_val'] = all_data["Cv"][:,1][:,np.newaxis].T

    # print("shapes are X_train:{}, y_train:{}, X_val:{}, y_val:{}"
    #       .format(data['X_train'].shape, data['y_train'].shape, data['X_val'].shape, data['y_val'].shape))
    image_size = data['X_train'].shape[0]
    num_labels = data['y_train'].shape[1]

    if test_type == "softmax":
        layer_shapes = [(image_size, num_labels)]
    elif test_type == "tanh":
        layer_shapes = [(image_size, 5),(5, num_labels)]
    elif test_type == "network":
        layer_shapes = [(image_size, hidden_layers_size)]
        num_hidden_layers -= 1 #not including the first hidden layer defined in the previous line
        layer_shapes += [(hidden_layers_size, hidden_layers_size)] * num_hidden_layers
        layer_shapes += [(hidden_layers_size, num_labels)]
        # layer_shapes = [(image_size, 25),(25, 50),(50, 50),(50, 25),(25, num_labels)]
    num_layers = len(layer_shapes)
    activations = ['tanh']
    loss_function = ['softmax']
    network = FC_Net(num_layers, layer_shapes, activations, loss_function, is_res)
    return data, network, layer_shapes


def get_layer_grad_info(grad_tester, layer_shapes, layers_wanted):

    layers_grad_info = dict()

    for param_layer in layers_wanted:
        layer_grad_info = dict()

        layer_grad_info['w_shape'] = (layer_shapes[param_layer - 1][1], layer_shapes[param_layer - 1][0])
        layer_grad_info['w2_shape'] = (layer_shapes[param_layer - 1][0], layer_shapes[param_layer - 1][1])
        layer_grad_info['b_shape'] = (layer_shapes[param_layer - 1][1], 1)

        d_w = np.random.randn(*layer_grad_info['w_shape'])
        layer_grad_info['d_w'] = d_w / np.linalg.norm(d_w)
        layer_grad_info['w'] = np.random.randn(*layer_grad_info['w_shape']) * 0.1

        d_w2 = np.random.randn(*layer_grad_info['w2_shape'])
        layer_grad_info['d_w2'] = d_w2 / np.linalg.norm(d_w2)
        layer_grad_info['w2'] = np.random.randn(*layer_grad_info['w2_shape']) * 0.1

        d_b = np.random.randn(*layer_grad_info['b_shape'])
        layer_grad_info['d_b'] = d_b / np.linalg.norm(d_b)
        layer_grad_info['b'] = np.random.randn(*layer_grad_info['b_shape']) * 0.1

        layer_grad_info['set_w'] = grad_tester.layers[str(param_layer)].set_w
        layer_grad_info['set_b'] = grad_tester.layers[str(param_layer)].set_b
        layer_grad_info['set_w2'] = grad_tester.layers[str(param_layer)].set_w2

        layers_grad_info[str(param_layer)] = layer_grad_info

    return layers_grad_info



def run_grad_tester(data, grad_tester, layer_shapes, test_type, is_res, iterations=10):

    batch_size = data['X_train'].shape[1]  # for a batch with all the data
    epochs = 1
    learning_rate = 0.1
    decrease_lr_time = 100
    decrease_lr_by = 2

    if test_type != "network":
        layers_wanted = [1] #the layer of the w for which we want to check the gradient
    else:
        layers_wanted = list(range(1,len(layer_shapes)+1))

    layers_grad_info = get_layer_grad_info(grad_tester, layer_shapes, layers_wanted)

    iterations = list(range(iterations))

    if is_res:
        params = ['W', 'b', 'W2']
    else:
        params = ['W', 'b']

    # params = ['W', 'W',  'W']

    for param_layer in layers_wanted:
        layer_grad_info = layers_grad_info[str(param_layer)]
        # print("param values are {}, {}, {}".format(grad_tester.layers[str(param_layer)].layer_info['W'][0,0],grad_tester.layers[str(param_layer)].layer_info['b'][0,0],grad_tester.layers[str(param_layer)].layer_info['W'][0,0]))
        for param in params:
            if not (param == 'W2' and param_layer == len(grad_tester.layers)):
                epsilon = 1
                original_param = np.copy(grad_tester.layers[str(param_layer)].layer_info[param])
                theta = layer_grad_info[param.lower()]
                layer_grad_info['set_'+param.lower()](theta)

                grad_tester.run_network(data, batch_size, epochs, learning_rate, decrease_lr_time, decrease_lr_by, gradient_testing=True)
                loss_0 = grad_tester.loss
                grad_theta = grad_tester.layers[str(param_layer)].layer_info['d'+param]

                res1 = []
                res2 = []

                run_iterations_for_grad_test(batch_size, data, decrease_lr_by, decrease_lr_time, epochs, epsilon, grad_tester,
                               grad_theta, iterations, layer_grad_info, learning_rate, loss_0, param, res1, res2, theta)
                plot_grad_graphs(iterations, layer_grad_info, original_param, param, param_layer, res1, res2, test_type)




def plot_grad_graphs(iterations, layer_grad_info, original_param, param, param_layer, res1, res2, test_type):
    if test_type != "network":
        print("-----------  parameter = {} ------------".format(param))
    else:
        print("-----------  {}  in layer {} ------------".format(param, param_layer))
    for i in range(len(iterations) - 1):
        print("Linear check should be 0.5 and is {:.2e}".format(res1[i + 1] / res1[i]))
    print("-----------------------------------------")
    for i in range(len(iterations) - 1):
        print("Quadratic check should be 0.25 and is {:.2e}".format(res2[i + 1] / res2[i]))
    plt.title("Gradient test for param {} in layer {}".format(param, param_layer))
    plt.semilogy(res1, 'g', label='Linear')
    plt.semilogy(res2, 'b', label='Quadratic')
    plt.legend(loc='upper right')
    plt.show()
    # print("changing {} from {} to {}".format(param, grad_tester.layers[str(param_layer)].layer_info[param][0,0], original_param[0,0]))
    layer_grad_info['set_' + param.lower()](original_param)
    # print("now is {}".format(grad_tester.layers[str(param_layer)].layer_info[param][0,0]))


def run_iterations_for_grad_test(batch_size, data, decrease_lr_by, decrease_lr_time, epochs, epsilon, grad_tester, grad_theta,
                   iterations, layer_grad_info, learning_rate, loss_0, param, res1, res2, theta):
    for _ in iterations:
        theta_i = theta + (epsilon * layer_grad_info['d_' + param.lower()])
        layer_grad_info['set_' + param.lower()](theta_i)
        grad_tester.run_network(data, batch_size, epochs, learning_rate, decrease_lr_time, decrease_lr_by,
                                gradient_testing=True)
        loss = grad_tester.loss

        res1.append(abs(loss - loss_0))
        num2 = abs(
            loss - loss_0 - epsilon * np.dot(layer_grad_info['d_' + param.lower()].flatten(), grad_theta.flatten()))
        res2.append(num2)
        epsilon = epsilon * 0.5


def w_jac_simple(tanh_grad, x, num_labels):
    lh = np.diag(tanh_grad.flatten())
    rh = np.kron(np.eye(num_labels), x.T)
    return np.dot(lh, rh)


def b_jac_simple(tanh_grad, x, num_labels):
    return np.diag(tanh_grad.flatten())

def w_jac_resnet(tanh_grad, x, num_labels):
    lh = np.diag(tanh_grad.flatten())
    rh = np.kron(np.eye(num_labels), x.T)
    return np.dot(lh, rh)

def b_jac_resnet(tanh_grad, x, num_labels):
    return np.diag(tanh_grad.flatten())


def jacobian_test(data_path, is_resnet=False):
    iterations = list(range(10))
    np.random.seed(7)
    all_data = scipy.io.loadmat(data_path)
    data = {}

    data['X_val'] = all_data["Yv"][:,1][:,np.newaxis]
    data['y_val'] = all_data["Cv"][:,1][:,np.newaxis].T

    image_size, num_images = data['X_val'].shape
    num_labels = data['y_val'].shape[1]

    W = np.random.randn(num_labels, image_size)
    b = np.random.randn(num_labels, 1)
    W2=None
    if is_resnet:
        W2 = np.random.randn(image_size, num_labels)

    params = [W, b]
    names = ['W', 'b']
    funcs = [w_jac_simple, b_jac_simple]

    activation = Tanh(image_size, num_labels, W, b, W2)

    setters = [activation.set_w, activation.set_b]

    fx = activation.forward(data['X_val'])['activation_output']

    tanh_output = activation.tanh_output

    tanh_grad = (1 - tanh_output**2).T

    for i in range(len(params)):
        param = params[i]
        func = funcs[i]
        name = names[i]
        setter = setters[i]

        jac_0 = func(tanh_grad, data['X_val'], num_labels)

        d = np.random.randn(*param.shape)

        epsilon = 0.4
        res1 = []
        res2 = []

        fx = fx.T

        for _ in iterations:
            epsilon *= 0.5
            v = (epsilon * d)

            jacobian_iter = np.dot(jac_0, v.flatten())

            param_i = param + v
            setter(param_i)
            activation.forward(data['X_val'])
            fx_i = activation.output.T

            res1.append(np.linalg.norm(fx_i - fx, 2))
            num2 = np.linalg.norm(fx_i - fx - jacobian_iter, 2)

            res2.append(num2)


        setter(param) #reset the param value
        fx = fx.T

        print("---------------------------  param = {}  ----------------------------".format(name))
        for i in range(len(iterations)-1):
           print("should be 0.5 and is {:.2e}".format(res1[i+1]/res1[i]))
        for i in range(len(iterations)-1):
           print("should be 0.25 and is {:.2e}".format(res2[i+1]/res2[i]))
        plt.title("Tanh jacobian test for param = {}".format(name))
        plt.semilogy(res1, 'g', label='Linear')
        plt.semilogy(res2, 'y', label='Quadratic')
        plt.legend(loc='upper right')
        plt.show()




def main():
    data_name = 'SwissRollData.mat'
    # data_name = 'PeaksData.mat'
    # data_name = 'GMMData.mat'
    data_path = '../data/'
    data_path = data_path + data_name
    # gradient_test(test_type="softmax", is_res=False, data_path=data_path, iterations=10)
    # gradient_test(test_type="tanh", is_res=False, data_path=data_path, iterations=10)
    # gradient_test(test_type="tanh", is_res=True, data_path=data_path, iterations=10)


    hidden_layers_size = 40
    num_hidden_layers = 4
    # gradient_test(test_type="network", is_res=False, data_path=data_path, iterations=10, num_hidden_layers=num_hidden_layers, hidden_layers_size=hidden_layers_size)
    # gradient_test(test_type="network", is_res=True, data_path=data_path, iterations=10, num_hidden_layers=num_hidden_layers, hidden_layers_size=hidden_layers_size)

    jacobian_test(data_path=data_path, is_resnet=True)

if __name__ == "__main__":
    main()

