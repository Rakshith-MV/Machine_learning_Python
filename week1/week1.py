import numpy as np
import matplotlib.pyplot as plt
x,y = np.loadtxt("ex1data1.txt",delimiter=",",usecols=(0,1),unpack = True)

def plot(x,y):
    plt.scatter(x,y)
    plt.show()

# plot(x,y)

one = np.ones((97,1))
X = np.column_stack((one , x))
[m,n] = list(( X.shape))

theta = np.zeros((n,1)).reshape(n,)

def CostFunction(x,t,y,m):
    sqError = np.square((np.matmul(x,t).reshape(97,)) - y) #reshape because it helps in converting a list of lists to a single list of numbers
    J = (sum(sqError)/(2*m))
    return J


def Gradient_Descent(x,y,m,t,iters,alpha):
    for i in range(iters):
        t = t - (alpha)* (X.transpose() @ ((X@t).reshape(97,) - y))/m
    return t
theta = (Gradient_Descent(X,y,m,theta,1500,0.01))
print(theta)

print([1, 3.5]@theta.transpose())