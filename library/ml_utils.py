"""
Utility functions for machine learning


Created Feb. 6, 2020
Matthew Hayman
mhayman@ucar.edu
"""

import numpy as np
import xarray as xr
import os

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

def ensure_path(path:str):
    """
    Checks if a path exists.  If it doesn't, creates the
    necessary folders.
    path - path to check for or create
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except FileExistsError:
                print()
                print('tried creating data directory but it already exists')
                print(path)
                print()
