from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


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
    paths=[]
    assert not filters % cardinality
    filters_per_group=filters//cardinality
    for i in range(cardinality):
        paths.append(Conv2D(filters_per_group, kernel_size=(3, 3), activation='relu',padding='same')(y))
    y=layers.Add()(paths)
    return y

def residual_block_b(y, cardinality,bottleneck_filters,filters_in, strides=(1, 1), project_shortcut=False):
    shortcut = y
    y=Conv2D(filters_width, kernel_size=(1, 1), activation='relu',padding='same')(y)
    y=grouped_convolution(y,filters_width, cardinality)
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
        y=residual_block_a(y,cardinality,bottleneck_filters,filters_in,strides=strides)
    return y
    

    
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
    return model