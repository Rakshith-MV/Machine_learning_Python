import numpy as np
from pprint import pprint

class layer_data():  # This is a data class, I've stored data of each hidden neural layer as attributes of the class
    def __init__(self,n1,n2):
        self.grad = np.zeros((n1,n2))
        self.bgrad = np.zeros((n2,))
        self.weights = np.random.randn(n1,n2)
        self.values = np.zeros((n1,)) 
        self.bias = np.random.randn(n2,)

class Network(layer_data):
    def __init__(self,layers,nodes,nin,nout): #Layers = int  || nodes,nin,nout are arrays
        self.layers = layers;    self.nodes = nodes  #Nodes will have info about only hidden layers
        self.nin = nin         
        self.nout  = nout
        self.data = ['n'+str(i) for i in range(layers)]
        self.sig = lambda x: 1/(1+np.exp(-x))
        self.output = None
        self.m,self.n = np.shape(nin)

    def sigmoid(self,other):
        h = np.array([i for i in map(self.sig,other)])
        return h
    
    def sigmoid_prime(self,other):
        h = np.multiply((self.sigmoid(other)),(1-self.sigmoid(other)))
        return h

#Cost funtion details -------------------------------------------------------------------------------------------------------------------------
    def cost_function(self):
        return np.sum(np.square(self.nout - self.output))*(1/self.m)
    
    def error(self):
        return 2*sum(sum(self.nout - self.output))

#---------------------------------------------------------------------------------------------------------------------------------------------------

    def initailize(self):     #initialize random weights 
        for j in (range(self.layers-1)):
            self.data[j] = layer_data(self.nodes[j],self.nodes[j+1])    #Elements of the list data will store the computational details of each layer except last on.
        self.data[0].bias = np.zeros((self.nodes[1],))                  #Changing bias term of the input layer, as we dont need to add bias term for the input layer.
        self.data[0].values = self.nin

#Gradient and weights are initialized as w0 would be weights for the mapping from 0 layer to 1st layer.
#we shall store only the value details for the last layer.

    def feed_forward(self):
        for i in range(1,self.layers-1):
            self.data[i].values = self.sigmoid(np.matmul(self.data[i-1].values,self.data[i-1].weights) + self.data[i-1].bias)
        self.output = self.sigmoid(self.data[-2].values@self.data[-2].weights + self.data[-2].bias)

#--------------------------------------------------------------------------------------------------------------------------------------------
    def update_values(self,alpha):
        for i in range(self.layers):
            self.data[i].weights = self.data[i].weights - (alpha/self.m)*self.data[i].grad

    def back_propagation(self): #cross entropy cost function is being used 
        self.data[-2].grad = self.error()*self.data[-2].weights
        self.data[-2].bgrad = self.error()*self.data[-2].bias
        print(self.data[-2].grad)
        for i in np.arange(-3,-(self.layers+1),-1):
            self.data[i].grad = np.multiply(self.data[i+1].grad,self.data[i].values)



    def display(self):
        for i in range(self.layers-1):
            print("---------------------------------------------------------------------")
            print(i)
            print("Values = ",end=' ')
            pprint(self.data[i].values)
            print("Weights = ",end=' ')
            pprint(self.data[i].weights)
            print("Gradient = ",end=' ')
            pprint(self.data[i].grad)
            print("Bias =",end=' ')
            pprint(self.data[i].bias)
            print("Bgrad =",end=' ')
            pprint(self.data[i].bgrad)
            print("---------------------------------------------------------------------")
        print("Output = ",end=' ')
        pprint(self.output)
        print("Main Error = ",end =' ')
        pprint(self.error())

    def display_shape(self):
        for i in range(self.layers-1):
            print("---------------------------------------------------------------------")
            print(i)
            print("Values = ",np.shape(self.data[i].values))
            print("Weights = ",np.shape(self.data[i].weights))
            print("Gradient = ",np.shape(self.data[i].grad))
            print("Bias =",np.shape(self.data[i].bias))
            print("Bgrad =",np.shape(self.data[i].bgrad))
            print("---------------------------------------------------------------------")
        print("Output = ",np.shape(self.output))
        print("Main Error = ",np.shape(self.error()))
