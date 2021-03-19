import torch
import numpy as np
from holodecml.FourierOpticsLib import *

class RAFT:
    
    def __init__(self, xsize, ysize):
        
        self.func_list = [np.abs, np.angle, np.real, np.imag]
        # define x (columns) and y (rows) coordinates
        self.ypix = np.arange(xsize)-xsize//2
        self.xpix = np.arange(ysize)-ysize//2
        self.rpix = np.sqrt(self.xpix[np.newaxis,:]**2 + self.ypix[:,np.newaxis]**2)
        
        # define the radial coordinate for the radial mean
        self.rad = np.arange(np.maximum(self.ypix.size//2,self.xpix.size//2))
        
    def radially_averaged_ft(self, image, im = 0):
        # FT the image and store the desired operations
        image_ft0 = OpticsFFT(image)  # FFT the image
        
        # define function for calculating radial mean
        avg_rad = lambda r, fun: fun(image_ft0[(self.rpix >= r - 0.5) & (self.rpix < r + 0.5)]).mean()
        
        def compute(func):
            image_ft_r_mean = np.vectorize(avg_rad)(self.rad, func)
            image_ft_r_mean[0] = image_ft_r_mean[0] / (image_ft_r_mean.size) # rescale DC term
            return image_ft_r_mean[np.newaxis,...]/255

        # calculate the radial mean of the FT
        return torch.FloatTensor(np.hstack(map(compute, self.func_list))[0])
    
# class RAFT:
    
#     def __init__(self, xsize, ysize):
        
#         self.func_list = [np.abs, np.angle, np.real, np.imag]
#         # define x (columns) and y (rows) coordinates
#         self.ypix = np.arange(xsize)-xsize//2
#         self.xpix = np.arange(ysize)-ysize//2
#         self.rpix = np.sqrt(xpix[np.newaxis,:]**2 + ypix[:,np.newaxis]**2)
        
#         # define the radial coordinate for the radial mean
#         self.rad = np.arange(np.maximum(self.ypix.size//2,self.xpix.size//2))
        
#     def radially_averaged_ft(self, image):
#         # FT the image and store the desired operations
#         image_ft0 = OpticsFFT(image)  # FFT the image
        
#         # define function for calculating radial mean
#         avg_rad = lambda r, fun: fun(image_ft0[(self.rpix >= r - 0.5) & (self.rpix < r + 0.5)]).mean()

#         # calculate the radial mean of the FT
#         image_ft_list = []
#         for func in self.func_list:
#             image_ft_r_mean = np.vectorize(avg_rad)(self.rad, func)
#             image_ft_r_mean[0] = image_ft_r_mean[0]/(image_ft_r_mean.size) # rescale DC term
#             image_ft_list+=[image_ft_r_mean[np.newaxis,...]/255]
#         if im == 0:
#             image_ft = da.array(np.concatenate(image_ft_list,axis=0)[np.newaxis,...])
#         else:
#             image_ft = da.concatenate([image_ft,np.concatenate(image_ft_list,axis=0)[np.newaxis,...]],axis=0)
        
#         return image_ft