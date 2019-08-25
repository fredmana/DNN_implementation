
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import scipy.io
import matplotlib.pyplot as plt
from layer import *

class FC_Net():

    def __init__(self, num_layers, layer_shapes, activations, loss_function, is_resnet):
        self.num_layers = num_layers
        self.layers = {}
        for i in range(num_layers-1):
            id = str(i+1)
            N, K = layer_shapes[i]
            self.layers[id] = Layer(id, N, K, activations, False, self.layers, is_resnet)
        last_id = str(num_layers)
        N, K = layer_shapes[num_layers-1] # shape for last layer params
        self.layers[last_id]= Layer(last_id, N, K, loss_function, True, self.layers)
        self.last_layer = self.layers[last_id]


    def run_network(self, data, batch_size, epochs, learning_rate, decrease_lr_time, decrease_lr_by, gradient_testing=False, verbose=False):
        self.info_per_epoch = self.train(data, batch_size, epochs, learning_rate, decrease_lr_time, decrease_lr_by, gradient_testing, verbose)
        self.test(data, gradient_testing, verbose)


    def train(self, data, bs, eps, lr, decrease_lr_time, decrease_lr_by, gradient_testing=False, verbose=False):
        X = data['X_train']
        y = data['y_train']
        info_per_epoch = {'train_accuracy':[], 'train_loss':[], 'test_accuracy':[], 'test_loss':[]}
        M = X.shape[1]

        for epoch in range(eps):

            #update the learning rate if needed.
            if (epoch+1) % decrease_lr_time == 0:
                lr /= decrease_lr_by
                print("Decreasing learning rate to {}".format(lr))

            if gradient_testing == False: #if we're not testing the gradient then shuffle the data
                X, y = self.suffle_data(X, y, M)

            #split the data to batches and run a single epoch
            batches = int(np.ceil(M / bs))
            for b in range(batches):

                Xb = X[:, b*bs:b*bs+bs]
                yb = y[b*bs:b*bs+bs, :]

                loss, accuracy = self.forward(Xb, yb)

                if b == batches -1: # in the last batch of every epoch save the train and test socres
                    info_per_epoch['train_accuracy'].append(accuracy)
                    info_per_epoch['train_loss'].append(loss)

                    test_loss, test_accuracy = self.test(data, verbose=False)
                    info_per_epoch['test_accuracy'].append(test_accuracy)
                    info_per_epoch['test_loss'].append(test_loss)

                if b % 250 == 0 and verbose == True:
                    print("For epoch = {}, batch {}/{}:\n\t loss is {} and accuracy is {} percent.".format(epoch+1, b, batches, loss, accuracy*100))
                if gradient_testing == False:
                    self.backprop(lr, gradient_testing)
        return info_per_epoch


    def forward(self, X, y):
        next_layer_input = X
        for layer_id in range(self.num_layers-1):
            layer = self.layers[str(layer_id+1)]
            layer.forward(next_layer_input)
            next_layer_input = layer.layer_info['activation_output']

        self.last_layer.forward(next_layer_input, y)
        self.loss = self.last_layer.layer_info['loss']
        self.accuracy = self.last_layer.layer_info['accuracy']
        return self.loss, self.accuracy


    def backprop(self, lr, gradient_testing=False):
        self.last_layer.backward(lr, input_grad=None, gradient_testing=gradient_testing)
        next_layer_grad_x = self.last_layer.layer_info['dX']
        for layer_id in reversed(range(0, self.num_layers-1)):
            layer = self.layers[str(layer_id+1)]
            layer.backward(lr, next_layer_grad_x, gradient_testing=gradient_testing)
            next_layer_grad_x = layer.layer_info['dX']


    def test(self, data, gradient_testing=False, verbose=False):
        X = data['X_val']
        y = data['y_val']
        loss, accuracy = self.forward(X, y)
        if gradient_testing==True: #Run backprop for the test data without parameter update, just to get the gradient.
            self.backprop(0.1, gradient_testing=gradient_testing)
        if verbose == True:
            print("\n########################################")
            print("For the test data:\nFinal loss for test data is {} and accuracy is {} percent.".format(loss, accuracy*100))
            print("########################################")
        return loss, accuracy


    def suffle_data(self, X, y, M):
        #shuffle x and y before splitting the data to bathces
        shuffle_order = np.arange(M)
        np.random.shuffle(shuffle_order)
        X = X[:,shuffle_order]
        y = y[shuffle_order,:]
        return X, y

    def show_epoch_performance(self):
        names = ['loss', 'accuracy']
        types = ['train', 'test']
        for name in names:
            plt.title("Train and test {} per epoch".format(name))
            results = []
            for type in types:
                results.append(self.info_per_epoch[type+'_'+name])
            plt.plot(results[0], 'r', label='train')
            plt.plot(results[1], 'y', label='test')
            plt.legend(loc='upper right')
            plt.show()
            # print('The train {} list is {}\nThe test {} list is {}'.format(name, results[0], name, results[1]))

def prepare_net_and_run(data_path, hidden_layers_size, num_hidden_layers, epochs, batch_size, is_resnet=False):
    all_data = scipy.io.loadmat(data_path)
    data = {}
    data['X_train'] = all_data["Yt"]
    data['y_train'] = all_data["Ct"].T
    data['X_val'] = all_data["Yv"]
    data['y_val'] = all_data["Cv"].T
    print("shapes are X_train:{}, y_train:{}, X_val:{}, y_val:{}"
          .format(data['X_train'].shape, data['y_train'].shape, data['X_val'].shape, data['y_val'].shape))
    image_size = data['X_train'].shape[0]
    num_labels = data['y_train'].shape[1]
    # data['y_train'] = np.argmax(data['y_train'], 0)[:,np.newaxis] #for testing sigmoid loss function
    # data['y_val'] = np.argmax(data['y_val'], 0)[:,np.newaxis] #for testing sigmoid loss function
    # layer_shapes = [(image_size, image_size), (image_size,image_size), (image_size,image_size), (image_size, num_labels)]
    # layer_shapes = [(image_size, image_size), (image_size,image_size), (image_size,image_size), (image_size,image_size), (image_size,image_size), (image_size,image_size), (image_size,image_size), (image_size,image_size), (image_size,image_size), (image_size,image_size), (image_size,image_size), (image_size,image_size), (image_size,image_size), (image_size,image_size), (image_size,image_size), (image_size,image_size), (image_size, num_labels)]
    # layer_shapes = [(image_size, 5),(5, 3),(3, num_labels)]
    # layer_shapes = [(image_size, 25),(25, 50),(50, 50),(50, 25),(25, num_labels)]
    # layer_shapes = [(image_size, 25),(25, 50),(50, 50),(50, 25),(25, 25),(25, 25),(25, 25),(25, 25),(25, 25),(25, 25),(25, num_labels)]
    layer_shapes = [(image_size, hidden_layers_size)]
    num_hidden_layers -= 1 #not including the first hidden layer defined in the previous line
    layer_shapes += [(hidden_layers_size, hidden_layers_size)] * num_hidden_layers
    layer_shapes += [(hidden_layers_size, num_labels)]
    num_layers = len(layer_shapes)
    activations = ['tanh']
    # activations = ['relu']
    loss_function = ['softmax']
    network = FC_Net(num_layers, layer_shapes, activations, loss_function, is_resnet)

    # batch_size = data['X_train'].shape[1] #for a batch with all the data
    learning_rate = 0.05
    decrease_lr_time = 100
    decrease_lr_by = 2
    network.run_network(data, batch_size, epochs, learning_rate, decrease_lr_time, decrease_lr_by, verbose=True)
    return network


def main():
    # data_path = '/Users/Asher/Desktop/my_Documents/MSc/courses/optimization/Project_2019/data/'
    data_path = '../data/'
    data_path = data_path + 'SwissRollData.mat'
    # data_path = data_path + 'PeaksData.mat'
    # data_path = data_path + 'GMMData.mat'

    hidden_layers_size = 40
    num_hidden_layers = 34
    batch_size = 32
    epochs = 50
    prepare_net_and_run(data_path, hidden_layers_size, num_hidden_layers, epochs, batch_size)


if __name__ == "__main__":
    main()

