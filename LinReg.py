import numpy as np
import matplotlib.pyplot as plt

class linear_models():
    def __init__(self,input,output,theta,m,n):
        self.i = input
        self.o = output
        self.t = theta
        self.m = m
        self.n = n

    def normal(self):
        return np.linalg.pinv(self.i)*np.transpose(self.t)
    def sigmoid(self):
        return 1/(1 + np.exp(-np.matmul(self.i,self.t)))

# Linear regression
#----------------------------------------------------------------------------------------------------#
    def squared_cost(self):
        error =  np.square(np.matmul(self.i,self.t) - self.o)
        return sum(error)/(2*self.m)                                    #Squared error
    
    def gradient_descent_squared(self):
        grad = (np.matmul(self.i,self.t) - self.o)
        grad = np.matmul(np.transpose(self.i),grad)/self.m
        return grad                                                     #compute gradient        

    def Linear_Regression(self,*,alpha,iterations):
        for i in range(iterations):
            self.t = self.t - alpha*self.gradient_descent_squared()     #updating the value of theta(parameter)
        return self.t
    
#Logistic regression
#------------------------------------------------------------------------------------------------------------#
    def cross_entropy_cost(self):
        error =  (np.matmul(self.o,np.log(np.transpose(self.sigmoid()))) - np.matmul(1-self.o,np.transpose(np.log(1-self.sigmoid()))))
        return (-1/self.m)*error

    def gradient_cross_entropy(self):
        grad = (-1/self.m)*(np.matmul(np.transpose(self.i),self.sigmoid()) - self.o)
        return grad

        
    def Logistic_Regression(self,*,alpha=0.01,iterations=1500):
        for i in range(iterations):
            self.t = self.t - alpha*self.gradient_cross_entropy()

#-----------------------------------------------------------------------------------------------------------------



class Data_and_Visualization:
    def __init__(self,input,output,theta,m,n):
        self.i = input
        self.o = output
        self.t = theta
        self.m = m
        self.n = n
    def generalization(self):
        for col in range(self.n):
            mu = np.mean(self.i[:,col])
            sigma = np.std(self.i[:,col])
            self.i[:,col] = (self.i[:,col] - mu)
            self.i[:,col] = sigma
        return self.i
    