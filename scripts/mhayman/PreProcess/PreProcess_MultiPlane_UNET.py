"""
Preprocess raw UNET training data
to provide multi-plane reconstruction
as the input channels.

This script is intended to follow on
Generate_UNET_Hologram.py
"""
import sys
import os
import numpy as np
import xarray as xr
import datetime
import copy

import FourierOpticsLib as FO

# set path to local libraries
dirP_str = '../../../library'
if dirP_str not in sys.path:
    sys.path.append(dirP_str)

import ml_utils as ml

ds_path = "/scr/sci/mhayman/holodec/holodec-ml-data/UNET/"
ds_file = "UNET_image_256x256_5000count_5particles_v02.nc"

rescale = 255

print()
print('loading file')
print(ds_file)
print('from path')
print(ds_path)

# specify number of layers to reconstruct
params = {'zplanes':10,
          'preprocess_type':'multi-plane reconstruction',
          'raw_file':ds_file,
          'complevel':9}

ds_fn = ds_file.split('_v')
save_file = ds_fn[0]+'_%dzplanes_v'%params['zplanes']+ds_fn[1]

save_path = ds_path + ds_file.replace('.nc','/')
if not os.path.exists(save_path):
    try:
        os.makedirs(save_path)
    except FileExistsError:
            print('tried creating data directory but it already exists')
            print(save_path)
            print()

print('new preprocess save file')
print(save_file)
print('in path')
print(save_path)
print()


ds0 = xr.open_dataset(ds_path+ds_file)
depth_array = np.linspace(ds0.attrs['zmin'],ds0.attrs['zmax'],params['zplanes'])

# create grid for reconstructing the data
grid2 = FO.Coordinate_Grid(((ds0.attrs['ydim'],ds0.attrs['xdim'],),
                           (ds0.attrs['pixel_width'],ds0.attrs['pixel_width'],))
                           ,inputType='ccd')

# depth is actually a combination of real and imaginary components along
# the z axis
channel = xr.DataArray(np.arange(depth_array.size*2),
                                dims=('channel'),
                                attrs={'description':'input channel index'})

image_planes = xr.DataArray(np.zeros((ds0.dims['hologram_number'],
                        ds0.dims['xsize'],ds0.dims['ysize'],channel.size),dtype=float),
                dims=('hologram_number','xsize','ysize','channel'),
                coords={'xsize':ds0.coords['xsize'],'ysize':ds0.coords['ysize'],'channel':channel})

channel_depth = xr.DataArray((depth_array[:,np.newaxis]*np.ones((1,2))).flatten(),
                                dims=('channel'),
                                attrs={'description':'z position of channel',
                                        'units':'meters'})
channel_type = xr.DataArray(['real','imag']*depth_array.size,
                                dims=('channel'),
                                attrs={'description':'defines if the channel represents a real or imaginary component'})

# iterate through each hologram and reconstruct the z-planes
for iholo in range(ds0.dims['hologram_number']):
    # initialize the reconstruction with the hologram
    E2 = FO.Efield(ds0.attrs['wavelength'],grid2,z=ds0.attrs['zmax'],fielddef=ds0['image'].isel(hologram_number=iholo).values/rescale)
    # reconstruct each depth plane
    for idepth,depthr in enumerate(depth_array):
        E2.propagate_to(depthr)
        image_planes.loc[{'hologram_number':iholo,'channel':2*idepth}] = np.real(E2.field)
        image_planes.loc[{'hologram_number':iholo,'channel':2*idepth+1}] = np.imag(E2.field)

    # report progress
    print(f"\r{iholo+1} of {ds0.dims['hologram_number']} holograms completed",end='')

# store the attributes from the raw file
# and some new ones
save_attr = copy.deepcopy(ds0.attrs)
for new_attr in params:
    save_attr[new_attr] = params[new_attr]

ds_pre = xr.Dataset({'xsize':ds0['xsize'],'ysize':ds0['ysize'],
                'image_planes':image_planes,
                'labels':ds0['labels'],'channel':channel,
                'channel_depth':channel_depth,
                'channel_type':channel_type},
                attrs=save_attr)



print("saving data with compression level %d to"%params['complevel'])
print(save_path+save_file)

nccomp = dict(zlib=True, complevel=params['complevel'],shuffle=True)
ncencoding = {var: nccomp for var in ds_pre.data_vars}
ds_pre.to_netcdf(save_path+save_file, encoding=ncencoding,format='netCDF4',engine='netcdf4')

ds0.close()

print('save complete')
