from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,Activation
from layers import SpatialTransformer
import numpy as np


def simple_conv(input_shape,num_classes,filters=64,frozen_layers=[]):
    model = Sequential()
    model.add(Conv2D(filters, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape,name='conv1',trainable="conv1" not in frozen_layers))
    model.add(Conv2D(filters*2, kernel_size=(3, 3),strides=(2,2),
                     activation='relu',
                     input_shape=input_shape,name='conv11',trainable="conv11" not in frozen_layers))
    model.add(layers.BatchNormalization())
    model.add(Conv2D(filters*4, (3, 3),strides=(2,2), activation='relu',name='conv21',trainable="conv21" not in frozen_layers))
    model.add(layers.BatchNormalization())
    model.add(Conv2D(filters*8, (3, 3),strides=(2,2), activation='relu',name='conv22',trainable="conv22" not in frozen_layers))
    model.add(layers.BatchNormalization())
    model.add(Flatten())
    model.add(Dense(filters*2, activation='relu',name='fc1',trainable="fc1" not in frozen_layers))
    model.add(layers.BatchNormalization())
    model.add(Dense(num_classes, activation='softmax',name='fc2',trainable="fc2" not in frozen_layers))
    return model


def downsampling(y,filters):
    y=Conv2D(filters, kernel_size=(3, 3), strides=(2,2),activation='relu')(y)
    return y
def batchrelu(y):
    y=layers.BatchNormalization()(y)
    y=layers.Activation("relu")(y)
    return y
    
from keras import layers
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization

def grouped_convolution(y,filters,cardinality):
    groups=[]
    assert not filters % cardinality
    filters_per_group=filters//cardinality
    for i in range(cardinality):
        start,end=(filters_per_group*i,filters_per_group*(i+1))
        group=layers.Lambda(lambda y: y[:,:,:,start:end])(y)
        group_y=Conv2D(filters_per_group, kernel_size=(3, 3), activation='relu',padding='same')(group)
        groups.append(group_y)
    y=layers.Concatenate()(groups)
    return y

def residual_block_b(y, cardinality,bottleneck_filters,filters_in, strides=(1, 1), project_shortcut=False):
    shortcut = y
    y=Conv2D(bottleneck_filters, kernel_size=(1, 1), activation='relu',padding='same')(y)
    y=grouped_convolution(y,bottleneck_filters, cardinality)
    y=Conv2D(filters_in, kernel_size=(1, 1), activation='relu',padding='same')(y)
    output=layers.Add()([y,shortcut])
    output = layers.BatchNormalization()(output)
    return output

def residual_block_a(y, cardinality,bottleneck_filters,filters_in, strides=(1, 1), project_shortcut=False):
    shortcut = y
    
    paths=[]
    assert not bottleneck_filters % cardinality
    filters_per_group=bottleneck_filters//cardinality
    for i in range(cardinality):
        path=Conv2D(filters_per_group, kernel_size=(1, 1), activation='relu',padding='same')(y)
        path=Conv2D(filters_per_group, kernel_size=(3, 3), activation='relu',padding='same')(path)
        path=Conv2D(filters_in, kernel_size=(1, 1), activation='relu',padding='same')(path)
        paths.append(path)
    y=layers.Add()(paths)
    
    
    y=layers.Add()([y,shortcut])
    y=layers.BatchNormalization()(y)
    return y

def residual_layer(y,blocks,cardinality,bottleneck_filters,filters_in,strides=(1,1)):
    for i in range(blocks):
        y=residual_block_b(y,cardinality,bottleneck_filters,filters_in,strides=strides)
    return y
    

# see https://github.com/titu1994/Keras-ResNeXt/blob/master/resnext.py
# https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce
def residual_network(input_shape,classes,cardinality,bottleneck_filters,initial_filters,frozen_layers=[]):
    image_tensor = layers.Input(shape=input_shape)
    y=image_tensor
    #conv1
    filters_in=initial_filters
    y=Conv2D(initial_filters, kernel_size=(3,3),padding='same')(y)
    y=batchrelu(y)

    #conv2
    y=residual_layer(y,3,cardinality,bottleneck_filters,filters_in)
    filters_in*=2
    y=downsampling(y,filters_in)
    
#     #conv3
    y=residual_layer(y,3,cardinality,bottleneck_filters,filters_in)
    filters_in*=2
    y=downsampling(y,filters_in)
    
#     #conv4
    y=residual_layer(y,3,cardinality,bottleneck_filters,filters_in)
#     y=downsampling(y,512)


    y = layers.GlobalAveragePooling2D()(y)
    y=Dense(classes, activation='softmax',name='fc2',trainable="fc2" not in frozen_layers)(y)
    
    model = keras.models.Model(inputs=[image_tensor], outputs=[y])
    model.name="residual_network"
    return model

def localization_resnext(input_shape,dense_n=50,initial_filters=64,filters_in=3,cardinality=8):
    dense_n=50
    
    
    image_tensor = layers.Input(shape=input_shape)
    
    y=Conv2D(initial_filters, kernel_size=(3,3),padding='same')(y)
    y=batchrelu(y)

    #conv2
    y=residual_layer(y,3,cardinality,bottleneck_filters,filters_in)
    filters_in*=2
    y=downsampling(y,filters_in)
    y = layers.GlobalAveragePooling2D()(y)
    y=Dense(dense_n)(y)
    y=Activation('relu')(y)
    
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((dense_n, 6), dtype='float32')
    weights = [W, b.flatten()]
    y=Dense(6, weights=weights)(y)
    
    model = keras.models.Model(inputs=[image_tensor], outputs=[y])
    model.name="residual_network"
    
    return model
    
def localization_network(input_shape,dense_n=50):
    dense_n=50
    b = np.zeros((2, 3), dtype='float32')
    b[0, 0] = 1
    b[1, 1] = 1
    W = np.zeros((dense_n, 6), dtype='float32')
    weights = [W, b.flatten()]
    locnet = Sequential()
    locnet.add(MaxPooling2D(pool_size=(2,2), input_shape=input_shape))
    locnet.add(Conv2D(20, (5, 5)))
    locnet.add(MaxPooling2D(pool_size=(2,2)))
    locnet.add(Conv2D(20, (5, 5)))

    locnet.add(Flatten())
    locnet.add(Dense(dense_n))
    locnet.add(Activation('relu'))
    locnet.add(Dense(6, weights=weights))
    
    return locnet

def stn_resnext(input_shape,classes,cardinality,bottleneck_filters,initial_filters,frozen_layers=[]):
    image_tensor = layers.Input(shape=input_shape)
    locnet=localization_network(input_shape)

    y=SpatialTransformer(localization_net=locnet, output_size=input_shape, input_shape=input_shape)(image_tensor)
    
    #conv1
    filters_in=initial_filters
    y=Conv2D(initial_filters, kernel_size=(3,3),padding='same')(y)
    y=batchrelu(y)

    #conv2
    y=residual_layer(y,3,cardinality,bottleneck_filters,filters_in)
    filters_in*=2
    y=downsampling(y,filters_in)
    
#     #conv3
    y=residual_layer(y,3,cardinality,bottleneck_filters,filters_in)
    filters_in*=2
    y=downsampling(y,filters_in)
    
#     #conv4
    y=residual_layer(y,3,cardinality,bottleneck_filters,filters_in)
#     y=downsampling(y,512)


    y = layers.GlobalAveragePooling2D()(y)
    y=Dense(classes, activation='softmax',name='fc2',trainable="fc2" not in frozen_layers)(y)
    
    model = keras.models.Model(inputs=[image_tensor], outputs=[y])
    model.name="stn_resnext"
    return model

def stn(input_shape,classes):
    locnet=localization_network(input_shape)

    model = Sequential()

    model.add(SpatialTransformer(localization_net=locnet, output_size=input_shape, input_shape=input_shape))

#     model.add(Conv2D(32, (3, 3), padding='same',input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))

    model.add(Dense(classes))
    model.add(Activation('softmax'))
    model.name="stn"
    return model