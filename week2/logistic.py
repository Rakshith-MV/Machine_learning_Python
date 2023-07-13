import numpy as np
import matplotlib.pyplot as plt

#Load data
x1,x2,y = np.loadtxt("ex2data1.txt",delimiter=",",usecols=(0,1,2),unpack=True)
X = np.column_stack((x1,x2))


#Plot the given data
for i,j,n in zip(x1,x2,range(len(x1))):
    if y[n] == 1:
        plt.plot(i,j,'+',color = 'k')
    else:
        plt.plot(i,j,'o',color='r')
plt.show()

def sigmoid(x):
    return 1/(1+np.exp(-x))