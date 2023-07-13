import numpy as np

x1,x2,y = np.loadtxt("ex1data2.txt",delimiter=',',usecols=(0,1,2),unpack=True)

m = len(x1)

X = np.column_stack((x1,x2))
n = np.shape(X)[1]

def generalize(x):
    mu = np.mean(x)
    s = np.std(x)
    x = (x - mu)/s
    return x

X[:,0] = generalize(X[:,0])
X[:,1] = generalize(X[:,1])

print(X)
X = np.column_stack((np.ones((m,1)),X))
n+=1
theta = np.zeros((n,))


def CostFunction(X,y,theta,m):
    sqError = np.square(X@theta.transpose() - y)
    J = sum(sqError)/(2*m)
    return J



def Gradient_Descent(x,y,theta,m,alpha,iters,n):
    for i in range(iters):
        grad =  np.zeros(n,)
        for j in range(m):
            for k in range(n):
                grad[k] = grad[k] + (theta[k]*x[j][k] - y[j])*x[j][k] 
        theta = theta - alpha*grad/m
    return theta


theta = (Gradient_Descent(X,y,theta,m,0.01,400,n))
print(theta)
print(theta.transpose() @np.array([1 ,1650 ,3]))
