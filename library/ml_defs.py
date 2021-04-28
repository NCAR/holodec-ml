"""
Functions for defining machine learning
architectures


Created Mar. 13, 2020
Matthew Hayman
mhayman@ucar.edu
"""

import numpy as np

from tensorflow.keras.layers import Activation, MaxPool2D, SeparableConv2D, concatenate, Conv2DTranspose,Flatten,Lambda, Conv2D,Dense, Reshape
import tensorflow.keras.backend as K

from typing import Tuple, List, Union



def add_unet_layers(input_node,n_layers,n_filters,nConv=5,
            nPool=4,activation="relu",kernel_initializer = "he_normal",cat=True):
    """
    Recursive function for defining a encoding/decoding UNET
    input_node - the input supplied to the UNET
    n_layers - the number of desired layers in the UNET
    n_filters - the number of convolution filters in the first layer
        this will grow by factors of two for each additional layer depth
    nConv - number of points in each convolution kernel
    nPool - number of points in each max-pool operation
    activation - activation function to use.  Typically 'relu'.
    cat - concatenate the feedforward onto the other side of the UNET

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
        if cat:
            concat_1u = concatenate([upsamp_1u,act_2d],axis=3)
        else:
            concat_1u = upsamp_1u

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

def add_unet_dense(input_node,n_layers,n_filters,nConv=5,
            nPool=4,activation="relu",kernel_initializer = "he_normal",
            Ndense=64,):
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
        conv_1d = Conv2D(n_filters, (nConv, nConv), padding="same", kernel_initializer = kernel_initializer)(input_node)
        act_1d = Activation("relu")(conv_1d)
        conv_2d = Conv2D(n_filters, (nConv, nConv), padding="same", kernel_initializer = kernel_initializer)(act_1d)
        act_2d = Activation("relu")(conv_2d)
        pool = MaxPool2D(pool_size=(nPool, nPool))(act_2d)

        # create the next layer below this one
        return_node = add_unet_dense(pool,n_layers-1,n_filters*2,nConv=nConv,nPool=nPool,activation=activation,Ndense=Ndense)

        # define the up sampling and feed foward layer
        upsamp_1u = Conv2DTranspose(n_filters, (nConv,nConv), strides=(nPool,nPool),padding="same")(return_node)
        concat_1u = concatenate([upsamp_1u,act_2d],axis=3)
        conv_1u = Conv2D(n_filters,(nConv,nConv),padding="same",kernel_initializer = kernel_initializer)(concat_1u)
        act_1u = Activation("relu")(conv_1u)
        conv_2u = Conv2D(n_filters,(nConv,nConv),padding="same",kernel_initializer = kernel_initializer)(act_1u)
        act_2u = Activation("relu")(conv_2u)
    else:
        # this is the bottom of the encoding layers
        channel_branch = []
        for i in range(n_filters//2):
            # Slicing the ith channel:
            chan = Lambda(lambda x: x[..., i])(input_node)
            chan_flat = Flatten()(chan)
            d1 = Dense(Ndense*Ndense,activation='tanh')(chan_flat)
            d2 = Dense(Ndense*Ndense,activation='tanh')(d1)
            d2_reshape = Reshape((Ndense,Ndense,1))(d2)
            channel_branch.append(d2_reshape)

        # Concatenating together the per-channel results:
        dense_out = concatenate(channel_branch,axis=3)

        conv_1 = Conv2D(n_filters,(nConv,nConv),padding="same", kernel_initializer = kernel_initializer)(dense_out)
        act_1 = Activation("relu")(conv_1)

        conv_2 = Conv2D(n_filters,(nConv,nConv),padding="same", kernel_initializer = kernel_initializer)(act_1)
        act_2u = Activation("relu")(conv_2)
    
    
    # return the result to the next layer up
    return act_2u

def add_unet_vae(input_node,n_layers,n_filters,nConv=5,
            nPool=4,activation="relu",kernel_initializer = "he_normal",
            latent_dim=64,n_dense_layers=2,):
    """
    Recursive function for defining a encoding/decoding UNET
    input_node - the input supplied to the UNET
    n_layers - the number of desired layers in the UNET
    n_filters - the number of convolution filters in the first layer
        this will grow by factors of two for each additional layer depth
    nConv - number of points in each convolution kernel
    nPool - number of points in each max-pool operation
    activation - activation function to use.  Typically 'relu'.
    latent_dim - number of neurons in the latent dense layers
    n_dense_layers - number of dense layers used to estimate the latent space 


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
        conv_1d = Conv2D(n_filters, (nConv, nConv), padding="same", kernel_initializer = kernel_initializer)(input_node)
        act_1d = Activation("relu")(conv_1d)
        conv_2d = Conv2D(n_filters, (nConv, nConv), padding="same", kernel_initializer = kernel_initializer)(act_1d)
        act_2d = Activation("relu")(conv_2d)
        pool = MaxPool2D(pool_size=(nPool, nPool))(act_2d)

        # create the next layer below this one
        return_node = add_unet_vae(pool,n_layers-1,n_filters*2,nConv=nConv,nPool=nPool,activation=activation,
                    latent_dim=latent_dim,n_dense_layers=n_dense_layers)

        # define the up sampling and feed foward layer
        upsamp_1u = Conv2DTranspose(n_filters, (nConv,nConv), strides=(nPool,nPool),padding="same")(return_node)
        # concat_1u = concatenate([upsamp_1u,act_2d],axis=3)  # feed forward - not implemented in VAE
        conv_1u = Conv2D(n_filters,(nConv,nConv),padding="same",kernel_initializer = kernel_initializer)(upsamp_1u)
        act_1u = Activation("relu")(conv_1u)
        conv_2u = Conv2D(n_filters,(nConv,nConv),padding="same",kernel_initializer = kernel_initializer)(act_1u)
        act_2u = Activation("relu")(conv_2u)
    else:

        input_shape = K.int_shape(input_node)
        zinput = Flatten()(input_node)
        z_mean = Dense(latent_dim,activation='relu')(zinput)
        z_log_var = Dense(latent_dim,activation='relu')(zinput)
        for _ in range(np.maximum(n_dense_layers-2,0)):
            # represent the mean and variance branches
            # with separate dense networks
            z_mean = Dense(latent_dim,activation='relu')(z_mean)
            z_log_var = Dense(latent_dim,activation='relu')(z_log_var)
        z_mean = Dense(latent_dim,activation='linear')(z_mean)
        z_log_var = Dense(latent_dim,activation='linear')(z_log_var)

        z = Lambda(vae_sample)([z_mean,z_log_var,latent_dim])

        x1 = Dense(np.prod(input_shape[1:]),activation='relu')(z)
        dense_out = Reshape(input_shape[1:])(x1)

        conv_1 = Conv2D(n_filters,(nConv,nConv),padding="same", kernel_initializer = kernel_initializer)(dense_out)
        act_1 = Activation("relu")(conv_1)

        conv_2 = Conv2D(n_filters,(nConv,nConv),padding="same", kernel_initializer = kernel_initializer)(act_1)
        act_2u = Activation("relu")(conv_2)
    
    
    # return the result to the next layer up
    return act_2u

def vae_sample(args):
    # random sample for VAE resampling latent space
    z_mean,z_log_var = args
    epsilon = K.random_normal(shape=K.shape(z_mean),
                    mean=0., stddev=1.)
    return z_mean + epsilon*K.exp(z_log_var)

def filtered_mae(y_true,y_pred):
    """
    Calculate mean absolute error for particle amplitude
    and position.

    Custom loss function for unet trained to identify
    particle images and positions.
    
    This function calculates mean absolute error of all
    amplitude pixels but calculates mean absolute error of
    only particle pixels on the z pixels.

    This assumes that the first channel is the z data and
    the second channel is the amplitude data.
    """
    # break out the truth values
    # assume z is 0 index and amplitude is 1 index
    z_true = y_true[...,0]
    a_true = y_true[...,1]

    # break out the predicted values
    z_pred = y_pred[...,0]
    a_pred = y_pred[...,1]

    # calculate mae amplitude loss
    a_loss = K.mean(K.abs(a_true-a_pred),axis=(1,2))

    # calculate mae z loss with masking
    z_loss = K.sum(K.cast(K.greater(a_true,0.10),'float32')*K.abs(z_true-z_pred),axis=(1,2))/K.maximum(K.sum(K.cast(K.greater(a_true,0.10),'float32'),axis=(1,2)),1)
    
    return a_loss+z_loss

def filtered_mse(y_true,y_pred):
    """
    Calculate mean square error for particle amplitude
    and position.

    Custom loss function for unet trained to identify
    particle images and positions.
    
    This function calculates mean absolute error of all
    amplitude pixels but calculates mean absolute error of
    only particle pixels on the z pixels.

    This assumes that the first channel is the z data and
    the second channel is the amplitude data.
    """

    # break out the truth values
    # assume z is 0 index and amplitude is 1 index
    z_true = y_true[...,0]
    a_true = y_true[...,1]

    # break out the predicted values
    z_pred = y_pred[...,0]
    a_pred = y_pred[...,1]

    # # normalization for mean
    # mean_norm = K.maximum(K.sum(K.cast(K.greater(a_true,0.10),'float32'),axis=(1,2)),1)

    # calculate mse amplitude loss
    a_loss = K.mean(K.square(a_true-a_pred),axis=(1,2))

    # calculate mse z loss with masking
    z_loss = K.mean(K.cast(K.greater(a_true,0.10),'float32')*K.square(z_true-z_pred),axis=(1,2))
    
    return a_loss+z_loss

def ks_test(y_true,y_pred):
    """
    Custom loss function for the
    Kolmogorov–Smirnov test.
    Requires that y is a 1D array representing
    a histogram
    """
    # return K.max(K.abs(K.cumsum(y_true,axis=1)-K.cumsum(y_pred,axis=1)))
    return K.mean(K.square(K.cumsum(K.cast(y_true,'float32'),axis=1)-K.cumsum(y_pred,axis=1)),axis=1)

def poisson_nll(y_true,y_pred):
    """
    negative log-likelihood loss function for
    Poisson observations
    """
    return K.sum(y_pred-y_true*K.log(y_pred+1e-9),axis=1)

def cum_poisson_nll(y_true,y_pred):
    """
    negative log-likelihood loss function for
    Poisson observations
    """
    return K.sum(K.cumsum(y_pred,axis=1)-K.cumsum(K.cast(y_true,'float32'),axis=1)*K.log(K.cumsum(y_pred,axis=1)+1e-9),axis=1)

def cum_mse(y_true,y_pred):
    """
    mean square error of CDF of output
    """
    return K.sum(K.square(K.cumsum(y_pred,axis=1)-K.cumsum(K.cast(y_true,'float32'),axis=1)),axis=1)

