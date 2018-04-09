from keras.datasets import mnist,fashion_mnist,cifar10

from keras import backend as K
import keras
import cluttered_mnist

def get_data(dataset="mnist"):
    # the data, shuffled and split between train and test sets
    if (dataset=="mnist"):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        img_channels,img_rows, img_cols = 1,28, 28
        labels=["0","1","2","3","4","5","6","7","8","9"]
        num_classes = 10
    elif dataset=="fashion_mnist":
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        img_channels,img_rows, img_cols = 1,28, 28
        labels=["tshirt","trouser","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle_boot"]
        num_classes = 10
    elif dataset=="cifar10":
        img_channels,img_rows, img_cols = 3,32,32
        labels=['dog', 'horse', 'frog', 'airplane', 'cat', 'ship', 'deer', 'bird', 'truck', 'automobile']
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes = 10
    elif dataset=="cluttered_mnist":
        (x_train, y_train), (x_test, y_test), (x_val, y_val),img_channels,img_rows, img_cols = cluttered_mnist.load_data()
        
        labels=["0","1","2","3","4","5","6","7","8","9"]
        num_classes = 10
    else:
        raise ValueError("Unknown dataset: %s" % dataset)
        
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], img_channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], img_channels, img_rows, img_cols)
        input_shape = (img_channels, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_channels)
        input_shape = (img_rows, img_cols, img_channels)
        
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test), input_shape,num_classes 