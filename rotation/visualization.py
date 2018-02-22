import keras.backend as K
import numpy as np

import matplotlib.pyplot as plt
    
def get_activations(model, model_inputs, layer_name=None):

    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]

    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
    return activations


def display_activations(activation_maps,layer_figsize=(10,15)):
    
    
    
    activations_n=len(activation_maps)
    
    #f, axarr = plt.subplots(activations_n, figsize=(40,70))
    
#     if activations_n==1:
#         axarr=[axarr]
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        f=plt.figure(figsize=layer_figsize)
        ax=f.gca(title='Layer %i' % i)
        ax.yaxis.set_ticks_position('none') 
        
        shape = activation_map.shape
        
        if len(shape) == 4:
            filters=activation_map.shape[-1]
            columns=15
            rows=filters//columns + (min(1,filters % columns))
            activations = make_mosaic_activations(activation_map[0],rows,columns,border=2)
            #np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            max_width=1024
            if num_activations > max_width:  # too hard to display it on the screen.
                height=num_activations//max_width + (min(1,num_activations % max_width))
                
                linear_activations = activations[0: max_width*height]
                
                activations=np.zeros((height,max_width))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
#         axarr[i].axis('off')
       
        nice_imshow(ax, activations)
        #axarr[i].imshow(activations, interpolation='None')
    plt.show()

    
def display_all_activations(model,sample):
    plt.imshow(sample[:,:,:])
    act=visualization.get_activations(rotated_model,[sample])
    visualization.display_activations(act)
    
import numpy.ma as ma

def make_mosaic_conv_weights(weights,border=1,columns=None):
    h,w,c,n = weights.shape
    reshaped_weights=weights.reshape((h,w,c*n))
    if columns==None:
        columns=c
    
    return make_mosaic_activations(reshaped_weights,0,columns)

        
def make_mosaic_activations(activations, rows, columns, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
        
    h,w,c = activations.shape
    if rows*columns<c:
        rows=c // columns + min(1,c % columns)
    
    
    mosaic = ma.masked_all((rows * h + (rows - 1) * border,
                            columns * w + (columns - 1) * border),
                            dtype=np.float32)
    
    paddedh = h + border
    paddedw = w + border
    for i in range(c):
        row = int(np.floor(i / columns))
        col = i % columns
        mosaic[row * paddedh:row * paddedh + h,
               col * paddedw:col * paddedw + w] = activations[:,:,i]
    return mosaic

from mpl_toolkits.axes_grid1 import make_axes_locatable

def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest')
    plt.colorbar(im, cax=cax)
    
import warnings
import math
def display_weights(model,columns=6):
    layers=model.layers[0].get_weights()[0]
    layers=len(model.layers)
    figure_width=20
    for i,layer in enumerate(model.layers):
        
        weights=layer.get_weights()
        if len(weights)==0:
           continue
        f=plt.figure()
        
        ax=f.gca()
        plt.title('Layer %d weights' % i)
        W=weights[0]
        print(W.shape)
        if len(W.shape)==4: # assume a convolutional node
            h,w,c,filters = W.shape
            if c==1:
                f.set_size_inches(figure_width/columns*filters, figure_width)
                columns=6
                rows=filters//columns + (min(1,filters % columns))
                mosaic=make_mosaic_activations(W[:,:,0,:],rows,columns)
                nice_imshow(ax,mosaic)
            else:
                f.set_size_inches(figure_width/columns*(math.sqrt(filters*c)), figure_width)
                nice_imshow(ax, make_mosaic_conv_weights(W))
        elif len(W.shape)==2:
            W=W.T
            nice_imshow(ax, W, cmap=plt.cm.binary)
            f.set_size_inches(figure_width*math.sqrt(W.shape[1])/W.shape[0], figure_width)
        elif (W.shape)==1:
            nice_imshow(ax, W, cmap=plt.cm.binary)
        else:
            warnings.warn("Displaying of weights with 3, 5 or >5 dimensions is not supported.")
        plt.show()
            
            
        
        
    
    
    