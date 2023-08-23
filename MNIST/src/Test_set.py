import imageio.v3 as iio
from PIL import Image
import numpy as np
import Neural_Networks_batch as nn

f = open("weights1.txt",'r')

#Flattening the weights and reshaping
z = f.readlines()
weights0= []
for i in z:
    l = i.lstrip("\n").lstrip("[").rstrip("\n").rstrip("]").lstrip("[").rstrip("]").split(" ")
    for k in l:
        try:
            weights0.append(float(k))
        except:
            None

weights0.pop()
weights1 = np.array(weights0[:50176]).reshape(64,784)
weights2 = np.array(weights0[50176:]).reshape(10,64)

# all the bias elements were same in this case, so just avoided complications by just copying
bias1 = np.ones((64,1))*-0.30058897
bias2 = np.ones((10,1))*-1.97035777e-16

#Loading test data
test = np.loadtxt("mnist_test.csv",delimiter=',',skiprows=1,dtype=np.longdouble)
test_data = test.T

def vectorize(Y):
    y = np.zeros((np.size(Y),10))
    for i in range(np.size(Y)):
        y[i][int(Y[i])] = 1
    return y.T
test_X = test_data[1:,:]/255
test_Y = test_data[0,:]

test_Y  = vectorize(test_Y)

net = nn.Network(test_X,test_Y,3,[784,64,10],[None,None],[test_X,test_Y],500)
net.initialize()

net.weights[0] = weights1
net.weights[1] = weights2

net.bias[0] = bias1
net.bias[1] = bias2

net.feed_forward(test_X)
a,l = (net.accuracy(True))

incorrect= test_X.T

#Didn't really know how to convert list to images, chatgpt came into the rescue;).
def list_to_image(pixel_list, width, height):
    if len(pixel_list) != width * height:
        raise ValueError("List length doesn't match the image dimensions")

    pixel_array = np.array(pixel_list, dtype=np.uint8)

    # Reshape the array into an image shape
    pixel_array = pixel_array.reshape((height, width))

    # Create an image from the NumPy array
    image = Image.fromarray(pixel_array, mode="L")  # "L" for grayscale

    return image

for r in l:
    image = list_to_image(incorrect[r]*255,28,28)
    image.save("C:/Users/Admin/OneDrive/Desktop/machine learning/code/errors/"+str(r)+"image.png")

#Rather than saving you could just see the image by image.show().
