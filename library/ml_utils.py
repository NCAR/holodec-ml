"""
Utility functions for machine learning


Created Feb. 6, 2020
Matthew Hayman
mhayman@ucar.edu
"""

import numpy as np
import xarray as xr


from typing import Tuple

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