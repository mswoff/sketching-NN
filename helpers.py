import pickle
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# data -- a 10000x3072 numpy array of uint 8
#  Each row of the array stores a 32x32 colour image. 
# The first 1024 entries contain the red channel values, 
# the next 1024 the green, and the final 1024 the blue. The 
# image is stored in row-major order, so that the first 32 
# entries of the array are the red channel values of the first 
# row of the image.
# labels -- a list of 10000 numbers in the range 0-9


   # Convert images from the CIFAR-10 format and
   #  return a 4-dim array with shape: [image_number, height, width, channel]
def load():
    ret = []
    labels = []
    for b_ind in range(5):
        batch = unpickle("cifar-10-batches-py/data_batch_" + str(b_ind+1))
        data = batch[b"data"]
        labels.append(np.array(batch[b'labels']))

        data = data.reshape([-1, 3, 32, 32])

        # Reorder the indices of the array.
        data = data.transpose([0, 2, 3, 1])

        ret.append(data)
    return ret, labels

def sigmoid(z):
    return 1./(1.+np.exp(-z))