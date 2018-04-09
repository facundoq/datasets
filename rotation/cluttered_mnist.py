"""CLUTTERED MNIST digits classification dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.utils.data_utils import get_file
from keras import backend as K
import numpy as np
import os

# see https://github.com/skaae/recurrent-spatial-transformer-code/blob/master/MNIST_SEQUENCE/create_mnist_sequence.py
# and https://github.com/MasazI/Spatial_Transformer_Network/blob/master/load_data.py

def load_data():
    """Loads the Cluttered MNIST dataset dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_val, y_val), (x_test, y_test)`.
    """
    dirname = 'cluttered-mnist-npz4'
#     origin = "https://github.com/skaae/recurrent-spatial-transformer-code/raw/master/mnist_sequence3_sample_8distortions_9x9.npz"
#     origin = 'https://github.com/daviddao/spatial-transformer-tensorflow/raw/master/data/mnist_sequence1_sample_5distortions5x5.npz'
    origin="https://s3.amazonaws.com/lasagne/recipes/datasets/mnist_cluttered_60x60_6distortions.npz"
    path = get_file(dirname, origin=origin)
    mnist_cluttered = np.load(path)
    

#     x_train = mnist_cluttered['X_train']
#     x_test = mnist_cluttered['X_test']
#     x_val = mnist_cluttered['X_valid']

    x_train = mnist_cluttered['x_train']
    y_train = mnist_cluttered['y_train'].argmax(axis=-1)


    x_test = mnist_cluttered['x_test']
    y_test = mnist_cluttered['y_test'].argmax(axis=-1)
    
    x_val = mnist_cluttered['x_valid']
    y_val = mnist_cluttered['y_valid'].argmax(axis=-1)
    
    img_channels,img_rows, img_cols = 1,60,60
    
    return (x_train, y_train), (x_test, y_test), (x_val, y_val),img_channels,img_rows, img_cols 