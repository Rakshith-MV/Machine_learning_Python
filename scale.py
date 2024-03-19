import numpy as np
from PIL import Image
def scale(arr:np.array, spec:chr = 'r', inverse:bool = False)->np.array:
    """
    Input :array of coloured image 
    Output : image in a particular scale
    """
    (a, b, c) = arr.shape
    new_image = np.zeros_like(arr)

    if spec == 'r':
        r  =  arr[:,:,0].flatten()
        if inverse == 1:
            r = 255 - r
        z = np.zeros_like(r)
        color = np.vstack((r, z, z)).T
        
    elif spec == 'g':
        r  =  arr[:,:,1].flatten()
        if inverse == 1:
            r = 255 - r
            
        z = np.zeros_like(r)
        color = np.vstack((z, r, z)).T
            
    elif spec == 'b':
        r  =  arr[:,:,2].flatten()
        if inverse == 1:
            r = 255 - r
            
        z = np.zeros_like(r)
        color = np.vstack((z, z, r)).T
        
    else:
        raise KeyError 
            
    for i in range(a):
        new_image[i] = (color[i*b:(i+1)*b])
    return new_image

#Pretty useless right now, we can just add resulting arrays from the above function

def multi_scale(arr, s:str, i=str)->np.array:  #Inverse tag isn't a default, input a string of binary values matching the scale
    new = scale(arr, s[0], i[0])
    for c in range(1,len(s)):
        new+=scale(arr,s[c],int(i[c]))
    return new