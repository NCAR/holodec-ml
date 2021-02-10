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


def calculate_d_moments(x_arr,y_arr,z_arr,d_arr,moments_arr=None,
                        zlim=None,circ_center=None,circ_radius=None):
    """
    Calculate the diameter moments for particles with
    corresponding x,y,z positions in a single hologram.  
    The particle positions are used to determine
    which particles are in the predefined sample volume.
    The sample volume can be constrained to a cylinder with user
    defined center, radius and depth


    inputs:
        x_arr : array of x positions of particles in the hologram
        
        y_arr : array of y positions of particles in the hologram
        
        z_arr : array of z positions of particles in the hologram
        
        d_arr : array of the particle diameters in the hologram
        
        moments_arr : array of the desired diameter moments.
            If not specified, uses 0-3.
        
        z_lim : list defining the maximum and minimum z values 
            to be included in the sample volume
            e.g. [z_min, z_max]
        
        circ_center : list or tuple of center of the sample volume 
            in the x-y plane [x_center, y_center]
            default does not filter particles based on x,y position
            e.g. set to [255,255] for 512 x 512 image grid
        
        circ_radius : radius of the sample volume in the x-y plane
            default does not filter particles based on x,y position
            e.g. set to 255 to use a circle inscribed in a 
            512x512 hologram

    returns:
        array corresponding to the evaluated diameter moments
    """

    # evaluated moments defaults to 0-3
    if moments_arr is None:
        moments_arr = np.arange(4)

    if zlim is not None:
        # identify particles contained in sample volume depth
        in_z = ( z_arr >= np.min(zlim) ) & ( z_arr <= np.max(zlim) )
    else:
        # default to assuming all particles are contained in
        # the sample volume depth
        in_z = np.ones(z_arr.shape, dtype=bool)

    if circ_center is not None and circ_radius is not None:
        in_r = ( x_arr - circ_center[0] )**2 + ( y_arr - circ_center[1] )**2 <= circ_radius**2
    else:
        in_r = np.ones(x_arr.shape, dtype=bool)
    
    in_particles_idx = np.where(in_z & in_r)

    diam_moments = np.sum(d_arr[in_particles_idx].reshape(1,-1) ** moments_arr.reshape(-1,1),axis=1)

    return diam_moments
    

