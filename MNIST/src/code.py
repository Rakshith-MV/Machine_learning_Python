import numpy as np
import matplotlib.pyplot as plt

"""
A Neural Network to classify digits(0-9), 
Still very unsure about the number of neurons to use, I'll start with a 3 layered neuron with one hidden layer 
I'll shall be using ReLu(Rectified linear unit) as the activation function(non linearity).
MNIST data set, 54000 training examples 6000 cross validation and 10000 test data.
"""

"""SOME INFORMATION
Data loaded
Weights =  (784, 15)                                    Each epoch 
Gradient =  (784, 15)                                           input = (6000,784)
Bgradient =  (1, 15)                                            output = (6000,)
Bias =  (1, 15)
Weights =  (15, 1)
Gradient =  (15, 1)
Bgradient =  (1, 1)
Bias =  (1, 1)
"""

#This class splits the data into batches and makes two arrays for training and cross_validation.
class Batches():
    def __init__(self,nin,length,batch_size):
        self.nin = nin
        self.length = length
        self.batch_size = batch_size
        self.training = []
        self.cross_validation,self.cross_validation_output = [],[]
        self.training_output = []
#Just fucking use the reshape funciton you moron!!
    def batch_split(self):
        np.random.shuffle(self.nin)
        epoch = int(self.length/self.batch_size)
        start = 0
        stop = self.batch_size
        new_data = []
        for i in range(epoch):
            if i != epoch-1:
                self.training.append(np.array(self.nin[start:stop,1:]))
                self.training_output.append(np.array(self.nin[start:stop,0]))
            else:
                self.cross_validation.append(np.array(self.nin[start:stop,1:]))
                self.cross_validation_output.append(np.array(self.nin[start:stop,0]))

            start += self.batch_size
            stop += self.batch_size

class Info():
    """
    This class will hold all the infomartion required, the batch-sets, weights, gradients,..
    the values are stored as arrays, i.e. weights[0] would be weights that would take from layer1 to layer 2 
    """
    def __init__(self,B,O,layers,neurons):
        self.set = B
        self.output = O
        self.weights = [np.random.randn(neurons[i],neurons[i+1]) for i in range(layers-1)]
        self.grad = [np.zeros((neurons[i],neurons[i+1])) for i in range(layers-1)]
        self.bias = [np.zeros((1,neurons[i+1])) for i in range(layers-1)]
        self.bgrad = [np.zeros((1,neurons[i+1])) for i in range(layers-1)]


class Network(Batches):
    def __init__(self,nin,length,layers,neurons):
        self.inn = nin
        self.size = length
        self.layers = layers
        self.neurons = neurons

    def initialize(self):           
        split_data = Batches(self.inn,self.size,6000)   #Get the data to split into cross-validation and training set
#!!!!!Work on batch size(6000)..
        split_data.batch_split()                          #split_data.cross_validation for cross_validation set
        data = Info(split_data.training,split_data.training_output,self.layers,self.neurons)    
        print(np.shape(data.set[0]))
        print(np.shape(data.output[0]))
    
#"-----------------------------------------------------------------------------------------------------------------------------------"
#"-----------------------------------------------------------------------------------------------------------------------------------"

