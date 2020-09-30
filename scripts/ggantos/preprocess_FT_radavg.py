"""
Preprocessing routine for histogram retrievals
This runs on the synthetic data Aaron generates.

This code is run from a script with the file,
settings and paths.

This code decimates a hologram by Fourier Transforming it
then performing a radial average to produce a 1D description
of the Hologram.

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
import scipy.signal

# set path to local libraries
sys.path.append('../../')

import holodecml.ml_utils as ml
import holodecml.FourierOpticsLib as FO



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
                'hist_edges':np.linspace(0,300,100),
                'max_hist_count':None
                }

"""

paths = {   'data':'/glade/scratch/mhayman/holodec/holodec-ml-data/',
            'save':'/glade/p/cisl/aiml/ggantos/holodec/ft_rad_bidis_z_realimag/'}

settings = {    'data_file':['synthetic_holograms_50-100particle_gamma_training.nc',
                            'synthetic_holograms_50-100particle_gamma_validation.nc',
                            'synthetic_holograms_50-100particle_gamma_test.nc'],
                'FourierTransform':True,
                'hist_edges': np.arange(0,200,10),# np.logspace(0,3,41),
                'max_hist_count':5000,
                'log_hist':False,
                'log_in':False
                }

func_list = [np.abs,np.angle,np.real,np.imag]
in_chan = ['abs','angle','real','imag']
# func_list = [np.abs,]
# in_chan = ['abs',]

histogram_edges = settings['hist_edges']
histogram_centers = 0.5*np.diff(histogram_edges) \
                    +histogram_edges[:-1]

run_date_str = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

if isinstance(settings['data_file'],str):
    data_file_list = [settings['data_file']]
else:
    # assume a list of data files were passed in
    data_file_list = settings['data_file']
    

for fn in data_file_list:

    # load the dataset file
    with xr.open_dataset(paths['data']+fn,chunks={'hologram_number':1}) as ds:
        # pre-process training data
        # generate a histogram for each image
        # initialize the particle property histogram bins

        if settings['max_hist_count'] is None:
            hologram_count = ds['hologram_number'].values.size # 5000, 1000, 1000
        else:
            hologram_count = settings['max_hist_count']

        if 'training' in fn:
            file_use = '_training_'
        elif 'validation' in fn:
            file_use = '_validation_'
        elif 'test' in fn:
            file_use = '_test_'
        else:
            file_use = '_'
        
        file_base = 'histogram'+file_use+'data_%dcount'%hologram_count+run_date_str

        print("   histogram bins: ")
        print("      "+str(histogram_centers.size))
        print("       ["+str(histogram_centers[0])+', '+str(histogram_centers[-1])+']') # '2.5, 192.5'
        print()
        
        print('   max particle size: %d'%ds['z'].values.max())
        print('   min particle size: %d'%ds['z'].values.min())
        print()

        # define x (columns) and y (rows) coordinates and calculate radial coordinate
        ypix = np.arange(ds.coords['xsize'].size)-ds.coords['xsize'].size//2 # 1200 x 0, ranges -400 to 399
        xpix = np.arange(ds.coords['ysize'].size)-ds.coords['ysize'].size//2 # 800 x 0, ranges -600 to 599
        rpix = np.sqrt(xpix[np.newaxis,:]**2+ypix[:,np.newaxis]**2) # 1200x800, ranges 0 to 721

        # define function for calculating radial mean
        avg_rad = lambda r,fun: fun(image_ft0[(rpix >= r-.5) & (rpix < r+.5)]).mean()
        # define the radial coordinate for the radial mean
        rad  = np.arange(np.maximum(ypix.size//2,xpix.size//2)) # 600 x 0, ranges 0 to 599

        # store the Fourier Transform and particle size histogram for each hologram
        print("Performing Fourier Transform")
        ft_start_time = datetime.datetime.now()
        for im in ds['hologram_number'].values[slice(None,settings['max_hist_count'])]:
            # find the particles in this hologram
            # hologram indexing is base 1
            particle_index = np.nonzero(ds['hid'].values==im+1)[0] # indices of particles for hologram in flat array of coordinates

            particle_count = ds['z'].values[particle_index].size # number of particles per hologram
            print(particle_count)
            # print(f'  found {particle_count} particles')

            h_moments = []
            for m in settings.get('moments',[0,1,2,3,4,5,6]):
                h_moments += [np.sum((ds['z'].values[particle_index]/2)**m)]
            h_moments = np.array(h_moments)

            # make a histogram of particles and store it in the data set
            # [  0,   5,  10,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,
            # 65,  70,  75,  80,  85,  90,  95, 100, 105, 110, 115, 120, 125,
            # 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190,
            # 195]
            hist0 = np.histogram(ds['z'].values[particle_index],
                        bins=histogram_edges)[0]
            if settings.get('log_hist',False):
                hist0 = np.log(hist0+1e-12)
            if im == 0:
                histogram = da.array(hist0[np.newaxis,...])
                histogram_moments = da.array(h_moments[np.newaxis,:])
            else:
                histogram = da.concatenate([histogram,hist0[np.newaxis,...]],axis=0)     
                histogram_moments = da.concatenate([histogram_moments,h_moments[np.newaxis,:]],axis=0)   
            
            if settings['FourierTransform']:
                # in_chan = list(settings['input_func'].keys())
                
                # FT the image and store the desired operations
                image0 = ds['image'].sel(hologram_number=im)  # select the hologram image, 1200 x 800
                image_ft0 = FO.OpticsFFT(image0)  # FFT the image, 1200 x 800
                
                # calculate the radial mean of the FT
                image_ft_list = []
                for func in func_list:
                    image_ft_r_mean = np.vectorize(avg_rad)(rad,func)
                    image_ft_r_mean[0] = image_ft_r_mean[0]/(image_ft_r_mean.size) # rescale DC term
                    image_ft_list+=[image_ft_r_mean[np.newaxis,...]/255]
                

                

                # perform requested operations for storage
                # image_ft_list = []
                # # for ik,func in enumerate(settings['input_func'].keys()):
                # #     image_ft_list+=[(settings['input_func'][func](image_ft0) / settings['input_scale'][func])[np.newaxis,...]]
                # #     # image_ft[func][im,:,:] = settings['input_func'][func](image_ft0) / settings['input_scale'][func]
                # if settings.get('log_in',False):
                #     image_ft_list = [np.log(1e-12+image_ft_r_mean)[np.newaxis,...]/np.log(255.0)]
                # else:
                #     image_ft_list = [image_ft_r_mean[np.newaxis,...]/255.0]
                
                if im == 0:
                    image_ft = da.array(np.concatenate(image_ft_list,axis=0)[np.newaxis,...]) # accumulating list of shape len x 1 x 600
                else:
                    image_ft = da.concatenate([image_ft,np.concatenate(image_ft_list,axis=0)[np.newaxis,...]],axis=0) # accumulating list of shape len x 1 x 600

            print(f'completed hologram {im} of {hologram_count}') # ,end='\r
        ft_stop_time = datetime.datetime.now()

        print('histogram shape:')
        print(histogram.shape)

        # if settings['n_decimate'] <= 1:
        #     xsize = ds.coords['xsize'].copy()
        #     ysize = ds.coords['ysize'].copy()
        # else:
        #     xsize = ds.coords['xsize'][settings['n_decimate']//2::settings['n_decimate']]
        #     ysize = ds.coords['ysize'][settings['n_decimate']//2::settings['n_decimate']]
        holo_num = ds.coords['hologram_number'].copy()
        image_dims = ds['image'].dims
        print('image dimensions')
        print(image_dims)
        print('image shape')
        print(ds['image'].shape)
        # print('xsize:%d'%xsize.size)
        # print('ysize:%d'%ysize.size)
        # if not settings['FourierTransform']:
        #     in_chan = ['real']
        #     image_ft = ds['image'].values[:,np.newaxis,...]
        # in_chan = ['abs']
        



    image_in_da = xr.DataArray(image_ft,
                                    coords={'hologram_number':holo_num[:hologram_count],
                                            'input_channels':in_chan,
                                            'rsize':rad},
                                    dims=[image_dims[0]]
                                        +['input_channels','rsize'])


    hist_bin_cent = xr.DataArray(histogram_centers,
                                    coords={'histogram_bin_centers':histogram_centers},
                                    dims=('histogram_bin_centers'))

    hist_bin_edges = xr.DataArray(histogram_edges,
                                    coords={'histogram_bin_edges':histogram_edges},
                                    dims=('histogram_bin_edges'))

    histogram_moments_da = xr.DataArray(histogram_moments,
                                    dims = ('hologram_number','moments'),
                                    coords={'hologram_number':holo_num[:hologram_count],
                                            'moments':settings.get('moments',[0,1,2,3,4,5,6])})

    histogram = histogram[...,np.newaxis]
    print('histogram shape')
    print(histogram.shape)
    histogram_da = xr.DataArray(histogram,
                dims=('hologram_number','histogram_bin_centers','output_channels'),
                coords={'hologram_number':holo_num[:hologram_count],
                        'histogram_bin_centers':hist_bin_cent,
                        'output_channels':['hist']})

    preproc_ds = xr.Dataset({'histogram':histogram_da,
                    'histogram_bin_centers':hist_bin_cent,
                    'histogram_bin_edges':hist_bin_edges,
                    'input_image':image_in_da,
                    'histogram_moments':histogram_moments_da},
                    attrs={'data_file':settings['data_file']})


    print("Writing to netcdf")
    if not os.path.exists(paths['save']):
        os.makedirs(paths['save'])
    print(paths['save']+file_base+".nc")
    preproc_ds.to_netcdf(paths['save']+file_base+".nc")

# # save the settings in human readable format
# # with a small file size
# json_dct = {'settings':settings,'paths':paths}
# for k in json_dct['settings']:
#     if hasattr(json_dct['settings'][k], '__call__'):
#         json_dct['settings'][k] = json_dct['settings'][k].__name__
#     if hasattr(json_dct['settings'][k],'__iter__'):
#         for j in range(json_dct['settings'][k]):
#             if hasattr(json_dct['settings'][k][j], '__call__'):
#                 json_dct['settings'][k][j] = json_dct['settings'][k][j].__name__
    
# with open(paths['save']+file_base+".json", 'w') as fp:
#     json.dump(json_dct, fp, indent=4)

print('write complete')