"""
Preprocessing routine for histogram retrievals
This runs on the synthetic data Aaron generates.

This code is run from a script with the file,
settings and paths.

created 7/28/2020
Matt Hayman
mhayman@ucar.edu
"""

# disable plotting in xwin
import matplotlib
matplotlib.use('Agg')

import sys
import os

import numpy as np
import xarray as xr
import dask.array as da
import datetime
import json


# set path to local libraries
dirP_str = '../../../library'
if dirP_str not in sys.path:
    sys.path.append(dirP_str)

import ml_utils as ml
import FourierOpticsLib as FO



"""
path to Aaron's synthetic data on linux share
/h/eol/bansemer/holodec/holodec-ml/datasets/synthetic_holograms_1particle_gamma_training.nc

expected inputs from calling script
paths = {   'data':'/glade/scratch/mhayman/holodec/holodec-ml-data/'
            'save':'/glade/scratch/mhayman/holodec/holodec-ml-data/histogram/'}

settings = {    'data_file':'synthetic_holograms_1particle_gamma_training.nc'
                'input_func':{'real':np.real,'imag':np.imag},
                'input_scale':{'real':255,'imag':255}
                'FourierTransform:True,
                'hist_edges':np.linspace(0,300,100)
                }

"""

histogram_edges = settings['hist_edges']
histogram_centers = 0.5*np.diff(histogram_edges) \
                    +histogram_edges[:-1]

file_base = 'histogram_training_data_'+datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

# load the dataset file
with xr.open_dataset(paths['data']+settings['data_file'],chunks={'hologram_number':1024}) as ds:
    # pre-process training data
    # generate a histogram for each image
    # initialize the particle property histogram bins

    print("   histogram bins: ")
    print("      "+str(histogram_centers.size))
    print("       ["+str(histogram_centers[0])+', '+str(histogram_centers[-1])+']')
    print()
    
    print('   max particle size: %d'%ds['d'].values.max())
    print()

    # store the Fourier Transform and particle size histogram for each hologram
    print("Performing Fourier Transform")
    ft_start_time = datetime.datetime.now()
    for im in ds['hologram_number'].values:
        # find the particles in this hologram
        # hologram indexing is base 1
        particle_index = np.nonzero(ds['hid'].values==im+1)[0]  
        
        # make a histogram of particles and store it in the data set
        hist0 = np.histogram(ds['d'].values[particle_index],
                    bins=histogram_edges)
        if im == 0:
            histogram = da.array(hist0[0][np.newaxis,...])
        else:
            histogram = da.concatenate([histogram,hist0[0][np.newaxis,...]],axis=0)        
        
        if settings['FourierTransform']:
            in_chan = list(settings['input_func'].keys())
            # FT the image and store the desired operations
            image0 = ds['image'].sel(hologram_number=im)  # select the hologram image
            image_ft0 = FO.OpticsFFT(image0-np.mean(image0))  # FFT the image
            # perform requested operations for storage
            image_ft_list = []
            for ik,func in enumerate(settings['input_func'].keys()):
                image_ft_list+=[(settings['input_func'][func](image_ft0) / settings['input_scale'][func])[np.newaxis,...]]
                # image_ft[func][im,:,:] = settings['input_func'][func](image_ft0) / settings['input_scale'][func]
            if im == 0:
                image_ft = da.array(np.concatenate(image_ft_list,axis=0)[np.newaxis,...])
            else:
                image_ft = da.concatenate([image_ft,np.concatenate(image_ft_list,axis=0)[np.newaxis,...]],axis=0)

    ft_stop_time = datetime.datetime.now()

    xsize = ds.coords['xsize'].copy()
    ysize = ds.coords['ysize'].copy()
    holo_num = ds.coords['hologram_number'].copy()
    image_dims = ds['image'].dims
    if not settings['FourierTransform']:
        in_chan = ['real']
        image_ft = ds['image'].values[:,np.newaxis,...]



image_in_da = xr.DataArray(image_ft,
                                coords={'hologram_number':holo_num,
                                        'input_channels':in_chan,
                                        'xsize':xsize,
                                        'ysize':ysize},
                                dims=[image_dims[0]]
                                    +[len(in_chan)]
                                    +list(image_dims[1:]))


hist_bin_cent = xr.DataArray(histogram_centers,
                                coords={'histogram_bin_centers':histogram_centers},
                                dims=('histogram_bin_centers'))

histogram_da = xr.DataArray(histogram,
            dims={'hologram_number','histogram_bin_centers'},
            coords={'hologram_number':holo_num,
                    'histogram_bin_centers':hist_bin_cent})

preproc_ds = xr.Dataset({'histogram':histogram_da,
                'histogram_bin_centers':hist_bin_cent,
                'histogram_bin_edges':histogram_edges,
                'input_image':image_in_da},
                attrs={'data_file':settings['data_file']})


print("Writing to netcdf")
preproc_ds.to_netcdf(paths['save']+file_base+".nc")

# save the settings in human readable format
# with a small file size
json_dct = {'settings':settings,'paths':paths}
with open(paths['save']+file_base+".json", 'w') as fp:
    json.dump(json_dct, fp, indent=4)

print('write complete')