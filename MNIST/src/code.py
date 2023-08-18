import numpy as np
import matplotlib.pyplot as plt

class Network():
    def __init__(self,nin,nout,layers,nodes,cv,test,batch_size:int):
        self.nin = nin
        self.m,self.n = np.shape(nin)   #m will be dimensions and n will be number of equations
        self.layers = layers
        self.nodes = nodes
        self.nout = nout
        self.batch = batch_size
        self.cross_validtion = cv
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
        return np.maximum(0,other)
    
    def softmax(self, Z):
        expZ = np.exp(Z- np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)

    def cost_function(self,start):
        return -np.mean(self.nout[:,start:start+self.batch]*np.log(self.activations[-1]+1e-8))
    
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
        return (self.activations[2] - self.nout[:,start:start+self.batch] )
    
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
            output = self.feed_forward(self.cross_validtion[0])
            t,runs = 0,0
            for i,j in zip(self.cross_validtion[1].T,self.activations[-1].T):
                z = input()
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
        cost = []
        val = []
        for k in range(iters):
            i = 0
            while(i<self.n):
                x = self.nin[:,i:i+self.batch]
                self.model(a,x,i)
                val.append(i)
                print(i)
                cost.append(self.cost_function(i))
                i+=self.batch
            
            if k%10 == 0:
                print(k,self.accuracy(False))            
