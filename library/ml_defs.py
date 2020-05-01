"""
Functions for defining machine learning
architectures


Created Mar. 13, 2020
Matthew Hayman
mhayman@ucar.edu
"""


from tensorflow.keras.layers import Activation, MaxPool2D, SeparableConv2D, concatenate, Conv2DTranspose
import tensorflow.keras.backend as K

from typing import Tuple, List, Union



def add_unet_layers(input_node,n_layers,n_filters,nConv=5,
            nPool=4,activation="relu",kernel_initializer = "he_normal"):
    """
    Recursive function for defining a encoding/decoding UNET
    input_node - the input supplied to the UNET
    n_layers - the number of desired layers in the UNET
    n_filters - the number of convolution filters in the first layer
        this will grow by factors of two for each additional layer depth
    nConv - number of points in each convolution kernel
    nPool - number of points in each max-pool operation
    activation - activation function to use.  Typically 'relu'.


    Example use:
    # UNET parameter definitions
    nFilters = 16
    nPool = 4
    nConv = 5
    nLayers = 4
    loss_fun = "mse"
    out_act = "linear" 

    # define the input based on input data dimensions
    cnn_input = Input(shape=scaled_in_data.shape[1:])  

    # create the unet
    unet_out = add_unet_layers(cnn_input,nLayers,nFilters,nConv=nConv,nPool=nPool,activation="relu")

    # add the output layer
    out = Conv2D(scaled_train_labels.sizes['type'],(1,1),padding="same",activation=out_act)(unet_out)

    # build and compile the model
    mod = Model(cnn_input, unet_out)
    mod.compile(optimizer="adam", loss=loss_fun,metrics=['acc'])
    mod.summary()

    """
    
    if n_layers > 1:
        # another layer will be created below this one
        
        # define the down sampling layer
        conv_1d = SeparableConv2D(n_filters, (nConv, nConv), padding="same", kernel_initializer = kernel_initializer)(input_node)
        act_1d = Activation("relu")(conv_1d)
        conv_2d = SeparableConv2D(n_filters, (nConv, nConv), padding="same", kernel_initializer = kernel_initializer)(act_1d)
        act_2d = Activation("relu")(conv_2d)
        pool = MaxPool2D(pool_size=(nPool, nPool))(act_2d)

        # create the next layer below this one
        return_node = add_unet_layers(pool,n_layers-1,n_filters*2,nConv,nPool,activation)

        # define the up sampling and feed foward layer
        upsamp_1u = Conv2DTranspose(n_filters, (nConv,nConv), strides=(nPool,nPool),padding="same")(return_node)
        concat_1u = concatenate([upsamp_1u,act_2d],axis=3)
        conv_1u = SeparableConv2D(n_filters,(nConv,nConv),padding="same",kernel_initializer = kernel_initializer)(concat_1u)
        act_1u = Activation("relu")(conv_1u)
        conv_2u = SeparableConv2D(n_filters,(nConv,nConv),padding="same",kernel_initializer = kernel_initializer)(act_1u)
        act_2u = Activation("relu")(conv_2u)
    else:
        # this is the bottom of the encoding layers
        conv_1 = SeparableConv2D(n_filters,(nConv,nConv),padding="same", kernel_initializer = kernel_initializer)(input_node)
        act_1 = Activation("relu")(conv_1)

        conv_2 = SeparableConv2D(n_filters,(nConv,nConv),padding="same", kernel_initializer = kernel_initializer)(act_1)
        act_2u = Activation("relu")(conv_2)
    
    
    # return the result to the next layer up
    return act_2u

def filtered_mae(y_true,y_pred):
    """
    Custom loss function for unet trained to identify
    particle images and positions.
    
    This function calculates mean absolute error of all
    amplitude pixels but calculates mean absolute error of
    only particle pixels on the z pixels.

    This assumes that the first channel is the z data and
    the second channel is the amplitude data.
    """

    z_true = y_true[...,0]
    a_true = y_true[...,1]

    z_pred = y_pred[...,0]
    a_pred = y_pred[...,1]

    # calculate mae amplitude loss
    a_loss = K.mean(K.abs(a_true-a_pred),axis=(1,2))

    # calculate mae z loss with masking
    z_loss = K.sum(K.cast(K.greater(a_true,0.10),'float32')*K.abs(z_true-z_pred),axis=(1,2))/K.sum(K.cast(K.greater(a_true,0.10),'float32'),axis=(1,2))
    
    return a_loss+z_loss