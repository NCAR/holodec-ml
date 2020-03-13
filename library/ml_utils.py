"""
Utility functions for machine learning


Created Feb. 6, 2020
Matthew Hayman
mhayman@ucar.edu
"""

import numpy as np
import xarray as xr

from tensorflow.keras.layers import Activation, MaxPool2D, SeparableConv2D, concatenate, Conv2DTranspose

from typing import Tuple, List, Union

class MinMaxScalerX:
    """
    Handles rescaling label data when working with
    xarray datasets.
    """
    def __init__(self,in_array: xr.DataArray,dim: Tuple=None):
        """
        Create a min/max scaler object that scales
        across dims so all data is in 0-1 range.  
        If dims is not provided, perform
        the scaling across all dims.
        Input is an xarray object

        inputs:
            in_array- xarray DataArray with label data
            dim - tuple listing the dimensions to perform
                scaling across.
                If not provided, rescaling is performed
                across all dimensions
        """
        self.min = in_array.min(dim=dim)
        self.max = in_array.max(dim=dim)
        self.delta = self.max-self.min
        self.mid = 0.5*(self.max+self.min)
    def fit_transform(self,new_array):
        """
        Apply rescaling to data
        """
        return (new_array-self.min)/self.delta
    
    def inverse_transform(self,out_array):
        """
        Invert rescaling to obtain original data
        """
        return out_array*self.delta+self.min

def next_pt(point:Union[Tuple,List],grid:np.ndarray,pgrow:np.int,decay:np.int=1.0):
    """
    recursively generates a random shape on a grid of zeros.
    inputs:
        point - current centroid of the shape (row,column) index
        grid - the grid of data to have the shape imposed on it
        pgrow - the probability that the shape grows
        decay - the decay in the growth with each recursive step
    """
    grid[point[0],point[1]] = 1
    pts_new = [[point[0]+1,point[1]],[point[0]-1,point[1]],[point[0],point[1]+1],[point[0],point[1]-1]]
    for pt in pts_new:
        if np.all(np.array(pt) >= 0) and np.all(np.array(pt) < np.array(grid.shape)):
            if np.random.rand() < pgrow and grid[pt[0],pt[1]] == 0:
                next_pt(pt,grid,pgrow*decay,decay=decay)

def add_unet_layers(input_node,n_layers,n_filters,nConv=5,nPool=4,activation="relu",kernel_initializer = "he_normal"):
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