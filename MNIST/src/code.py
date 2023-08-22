import numpy as np
import matplotlib.pyplot as plt
# A python model for recognition of hand written digits 0-9.
# Standart mnist data set, cosisting 60000 training data and 10000 test data.
# vectors of input stacked in columns. 
### Quick into to the inputs
#nin = input of 784 pixels of data as column vectors
#nout = conversion of an int(0-9) to a list of 10 elements with 1 to indicate the output. For example 3 would be [0,0,0,1,0,0,0,0,0,0].T
#layer = no. of layers
#nodes = number of neurons or nodes in each layer
#cd = list of inputs and outputs of cross validation set
#test = list of inputs and outputs of test set
#batch size = mini-batch gradient descent is implemented for efficient computations and convergence.
#additional inputs alpha and number of iterations to be mentioned while training
###

#One could also use this module to train for other problems too, with some basic modifications (But only three layers are including one hidden layer).


class Network():
    def __init__(self,nin:list,nout:list,layers:int,nodes:list,cv:list,test:list,batch_size:int):
        self.nin = nin
        self.m,self.n = np.shape(nin)   #m will be dimensions and n will be number of equations
        self.layers = layers
        self.nodes = nodes
        self.nout = nout
        self.batch = batch_size
        self.cross_validation = cv
        self.test_set = test
#------------------------------------------------------------------------------------------------
    def initialize(self):
        self.weights = [0.01*np.random.rand(self.nodes[i+1],self.nodes[i]) for i in range(self.layers-1)]
        self.grad = [np.zeros((self.nodes[i+1],self.nodes[i])) for i in range(self.layers-1)]
        self.bias = [np.zeros((self.nodes[i+1],1)) for i in range(self.layers-1)]
        self.bgrad = np.copy(self.bias)
        self.activations = list(range(self.layers))
        self.values = list(range(self.layers))


    def ReLu(self,other):
        return np.maximum(0,other)       #Rectified linear unit
    
    def softmax(self, Z):
        expZ = np.exp(Z- np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def cost_function(self,start):
        return -np.mean(self.nout[:,start:start+self.batch]*np.log(self.activations[-1]+1e-8))
    
    def cross_validation_error(self):
        return -np.mean(self.cross_validation[1]*np.log(self.activations[-1]+1e-8))
    
    def Deriv_Relu(self,x):
        d = np.zeros_like(x)
        d[x > 0] = 1
        return d
    
    def plot(self,x,y,x1,y1):
        plt.plot(x,y,xlabel=x1,ylabel=y1)
#-----------------------------------------------------------------------------------------------
    def feed_forward(self,x):
        self.activations[0] = x
        self.values[1] = self.weights[0].dot(self.activations[0])+self.bias[0]
        self.activations[1] = self.ReLu(self.values[1])
        self.values[2] = self.weights[1].dot(self.activations[1]) + self.bias[1]
        self.activations[2] = self.softmax(self.values[2])

    def error(self,start):
        return (self.activations[2] - self.nout[:,start:start+self.batch])
    
    def back_propagation(self,x,start):
        E = self.error(start)
        dz2 = E
        self.grad[1] = dz2.dot(self.activations[1].T)/self.batch
        self.bgrad[1] = np.sum(dz2)/self.batch
    
        dz1 = self.Deriv_Relu(self.values[1])*((self.weights[1].T).dot(dz2))  #derivative of relu of z1
        self.grad[0] = dz1.dot(self.activations[0].T)/self.batch
        self.bgrad[0] = np.sum(dz1)/self.batch

    def update(self,a):
        for j in range(self.layers-1):
            self.weights[j] -= a*self.grad[j]
            self.bias[j] -= a*self.bgrad[j]

    def model(self,a,x,start):   
        self.feed_forward(x)
        self.back_propagation(x,start)
        self.update(a)

    def accuracy(self, two):
        if two == False:
            output = self.feed_forward(self.cross_validation[0])
            t,runs = 0,0
            for i,j in zip(self.cross_validation[1].T,self.activations[-1].T):
                if np.argmax(i) == np.argmax(j):
                    t+=1
                runs+=1
            return t/runs
        else:
            output = self.feed_forward(self.test_set[0])
            t,runs = 0,0
            for i in zip(self.test_set[1].T,self.activations[-1].T):
                if np.argmax(i) == np.argmax(j):
                    t+=1
                runs +=1
            return t/runs


    def train(self,a,iters):
        cross_error = []
        train_error = []
        for k in range(iters):
            i = 0
            while(i<self.n):
                x = self.nin[:,i:i+self.batch]
                self.model(a,x,i)

                i+=self.batch
            
            if k%100 == 0:
                print(k,self.accuracy(False)) 
                plt.plot(k,self.accuracy(False))
            
            self.feed_forward(self.cross_validation[0])
            cross_error.append(self.cross_validation_error())
        
        a = np.arange(iters)
        plt.plot(a,cross_error,'r')
        plt.show()
