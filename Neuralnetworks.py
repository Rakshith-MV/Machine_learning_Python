import numpy as np

class layer_data():  # This is a data class, I've stored data of each neural layer as attributes of the class
	def __init__(self):
		self.grad = []
		self.weights =[] ; self.values = [] 


class Network(layer_data):
    def __init__(self,layers,nodes,nin,nout): #Layers = int  || nodes,nin,nout are arrays
        self.layers = layers;    self.nodes = nodes  # nodes doesn't just have info about hidden layers it composes of number of nodes in both input and ouput layers
        self.nin = nin         
        self.nout  = nout
        self.data = ['n'+str(i) for i in range(layers)]
        self.m,self.n = None   #Dimensions_clarification
        self.sig = lambda x: 1/(1+np.exp(-x))




        

    def sigmoid(self,other):
        h = [i for i in map(self.sig,other)]
        return h
    

#   Cost funtion details -------------------------------------------------------------------------------------------------------------------------
    def cost_function(self):
        return #add cross entropy cost function

#---------------------------------------------------------------------------------------------------------------------------------------------------

    def initailize(self):
        for i in self.data:
            i = layer_data
            if i != self.layers: i.weights = np.random.randn(self.nodes[i],self.nodes[i+1])
        for i in range(1,self.layers):
            self.data[i].values = self.sigmoid(np.matmul(self.data[i-1].values,self.data[i].weights))

    def feed_forward(self):
        for i in range(1,self.layers):
            self.data[i].values = self.sigmoid(np.matmul(self.data[i-1],self.data[i].weights))

#--------------------------------------------------------------------------------------------------------------------------------------------
    def update_values(self,alpha):
        for i in range(self.layers):
            self.data[i].values = self.data[i].values - alpha*self.data[i].grad
            self.data[i].grad = []

    def compute_gradient(self):  #sigmoid function
        pass                        #gradient i.e. the sensitivity of each node towards the output(or the cost function) is computed and fed into backpropagation
    

    def back_propagation(self): #cross entropy cost function is being used 
        self.data[-1].grad = self.data[-1].values - self.nout
        for i in np.arange(self.layers-2,-1):
            self.data[i].grad = self.data[i+1].grad
