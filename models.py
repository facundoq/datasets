#  original https://gist.github.com/mjdietzx/0cb95922aac14d446a6530f87b3a04ce

from keras import layers
from keras import models
from keras import constraints



def residual_network(x,classes,cardinality = 16):
    """
    ResNeXt by default. For ResNet set `cardinality` = 1 above.
    
    """
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        return y

    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y

    # conv1
    x = layers.Conv2D(32, kernel_size=(5, 5), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)

    # conv2
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(3):
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 32, 64, _project_shortcut=project_shortcut)

    # conv4
    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 64, 64, _strides=strides)
    
#     # conv5
#     for i in range(2):
#         strides = (2, 2) if i == 0 else (1, 1)
#         x = residual_block(x, 256, 512, _strides=strides)
    x=layers.Conv2D(4, (1, 1), padding='same', activation='relu', kernel_constraint=constraints.maxnorm(3))(x)
#     x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(classes, kernel_initializer='normal', activation='sigmoid')(x)
#     model.add(Flatten())
#     model.add(Dense(150, kernel_initializer='normal', activation='relu'))
#     model.add(Dense(classes, kernel_initializer='normal', activation='sigmoid'))

    return x

def resnext(input_shape,classes):
    image_tensor = layers.Input(shape=input_shape)
    network_output = residual_network(image_tensor,classes)

    model = models.Model(inputs=[image_tensor], outputs=[network_output])
    return model



def conv_mask(input_shape,classes):
    from keras.models import Sequential
    from keras.layers.convolutional import Conv2D
    from keras.layers.convolutional import MaxPooling2D
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Flatten
    from keras.layers import Input
    from keras.layers import BatchNormalization
    from keras.constraints import maxnorm
    from keras import layers
    def add_block(model,radius,filters,strides=1):
        model.add(Conv2D(filters, (radius, radius), padding='same', activation='relu',kernel_constraint=maxnorm(3),strides=(strides,strides))) 
        model.add(BatchNormalization())
        
    model = Sequential()
    model.add(Conv2D(3, (1,1), padding='same', activation='relu',kernel_constraint=maxnorm(3),input_shape=input_shape)) 
    
    for i in range(2):
        add_block(model,3,64)
    add_block(model,3,64,strides=2)
    for i in range(2):
        add_block(model,3,32)
    for i in range(2):
        add_block(model,3,16)
    for i in range(2):
        add_block(model,3,8)
    for i in range(2):
        add_block(model,3,4)
    for i in range(2):
        add_block(model,3,3)
    model.add(Conv2D(3, (1, 1), input_shape=input_shape, padding='same', activation='sigmoid'))
    model.add(Flatten())
              
    return model

def simple_conv(input_shape,classes):
    from keras.models import Sequential
    from keras.layers.convolutional import Conv2D
    from keras.layers.convolutional import MaxPooling2D
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Flatten
    from keras.layers import BatchNormalization
    from keras.constraints import maxnorm
    from keras import layers
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    for i in range(3):
        model.add(Conv2D(32, (3, 3), strides=(2,2), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
        model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), input_shape=input_shape, padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Conv2D(8, (3, 3), input_shape=input_shape, padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Conv2D(4, (3, 3), input_shape=input_shape, padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Flatten())
    model.add(Dense(150, kernel_initializer='normal', activation='relu'))
    model.add(Dense(classes, kernel_initializer='normal',activation='sigmoid'))
    #model.add(layers.GlobalAveragePooling2D())
    return model