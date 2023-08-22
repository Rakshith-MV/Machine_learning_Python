import numpy as np
import Neural_Networks_batch
data = np.loadtxt("mnist_train.csv",delimiter=',',skiprows=1,dtype=np.longdouble)

test = np.loadtxt("mnist_test.csv",delimiter=',',skiprows=1,dtype=np.longdouble)

test_data = test.T

np.random.shuffle(data)
data = data.T
X = data[1:,:54000]/255
Y = data[0,:54000]

CV_X = data[1:,54000:]/255
CV_Y = data[0,54000:]

test_X = test_data[1:,:]/255
test_Y = test_data[0,:]

def vectorize(Y):
    y = np.zeros((np.size(Y),10))
    for i in range(np.size(Y)):
        y[i][int(Y[i])] = 1
    return y.T

output = vectorize(Y)
CV_output = vectorize(CV_Y)
test_Y  = vectorize(test_Y)
print("Data loaded")

net = Neural_Networks_batch.Network(X,output,3,[784,64,10],[CV_X,CV_output],[test_X,test_Y],500)
net.initialize()
iters = 15000
alpha = 0.001
net.train(alpha,iters)

print(net.weights)
print(net.bias)
