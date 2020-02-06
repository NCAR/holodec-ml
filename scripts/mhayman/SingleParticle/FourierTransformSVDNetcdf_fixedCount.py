import numpy as np
import xarray as xr
import pandas as pd
import datetime

import FourierOpticsLib as FO
import MieLibrary as mie

ds_path = "/scr/sci/mhayman/holodec/holodec-ml-data/"
# ds_name = "synthethic_holograms_v0.nc"  # 1 particle data
ds_name = "synthetic_holograms_v02.nc"  # 3 particle data
# ds_name = "synthetic_holograms_v03.nc"  # 1-5 particle data
ds_base = ds_name.replace(".nc","")

# define the functions to be applied to the fourier transformed data
ft_func = {'amplitude':np.abs}
ft_scale = {'real':255,'imag':255,'amplitude':255}  # rescaling factors
encoded_dtype = "float"

particle_count = 3  # number of particle outputs expected for NN
part_components = ['d',] # particle parameters to train on

# define data needed for SVD
pixwid = 3e-6  # size of pixels
wavelength = 355e-9  # laser wavelength
num_components = 20  # number of principle components to use

print("Fourier Transforming: "+ds_name)
print("   path: "+ds_path)
print("   data type: "+encoded_dtype)
print("   output operations: ",end="")

with xr.open_dataset(ds_path+ds_name) as ds:
    # initialize the image Fourier Transform channels
    image_svd = {}
    for func in ft_func.keys():
        print(func,end=", ")
        image_svd[func] = xr.DataArray(np.zeros((num_components,ds['image'].coords['hologram_number'].size),dtype='float32'),
                            dims=['filter_number','hologram_number',])
    print()

    """
    Create SVD filters for the specified grid
    """
    print()
    print("Performing Singular Value Decomposition")
    image_grid = FO.Coordinate_Grid(((ds['image'].coords['xsize'].size,ds['image'].coords['ysize'].size),(pixwid,pixwid)),inputType='ccd')
    # Perform PCA to prefilter input data
    max_angle = np.sqrt(np.max(image_grid.fx**2+image_grid.fy**2))*wavelength
    ang_grid = np.linspace(0,max_angle,500)

    particle_range = np.linspace(5,100,100)*1e-6

    scat_data = np.zeros((ang_grid.size,particle_range.size))
    for ir,r in enumerate(particle_range):
        scat_data[:,ir] = np.abs(mie.Mie_PhaseMatrix(1.3,2*np.pi*r/wavelength,ang_grid)[0,:])
        scat_data[:,ir] = scat_data[:,ir]/np.sum(scat_data[:,ir])  # normalize the area under the curve

    pca_data = scat_data.copy()
    pca_mean = np.mean(pca_data,axis=0,keepdims=True)
    pca_data = pca_data-pca_mean

    u,s,v = np.linalg.svd(pca_data.T)
    itrunc = num_components
    vtrunc = v[:itrunc,:]
    utrunc = u[:,:itrunc]

    # create PCA informed filters
    grid_set = image_grid.fr.T*wavelength
    filter_set = xr.DataArray(np.zeros((ds["image"].sizes['xsize'],ds["image"].sizes['ysize'],itrunc)),
                        dims=('xsize','ysize','filter_number'))
    ft_shape = (ds["image"].sizes['xsize'],ds["image"].sizes['ysize'])
    for ai in range(itrunc):
        filter_set.loc[{'filter_number':ai}] = np.interp(grid_set.flatten(),ang_grid,vtrunc[ai,:]).reshape(ft_shape)
    
    # initialize particle training data
    particle_data = xr.DataArray(np.zeros((len(part_components),particle_count,ds['image'].sizes['hologram_number']),dtype=float),
                                dims=('particle_property','particle_number','hologram_number'),
                                coords={'particle_property':part_components})

    # use a sub-dataset for the particle data where
    # its sorted by particle size
    # hopefully this will help NN figure out what order to
    # place its outputs
    # perform in decending so missing data treated as zeros still 
    # makes sense in the sorting paradigm
    part_ds = ds[['hid']+part_components].sortby('d',ascending=False)


    print("")
    print("Applying SVD filters to FT data and Creating Histograms")
    print()
    ft_start_time = datetime.datetime.now()
    # create a variable with for each Fourier Transform operation of each image
    # to project onto each filter
    for im in range(ds['image'].sizes['hologram_number']):
        # sort the particle data by hologram number
        particle_index = np.nonzero(part_ds['hid'].values==im+1)[0]  # find particles associated with this hologram
        particle_data.loc[{'hologram_number':im,'particle_number':slice(None,len(particle_index))}]= \
                        part_ds[particle_data.coords['particle_property']].isel(particle=particle_index).to_array()

        image0 = ds['image'].isel(hologram_number=im).transpose('xsize','ysize')
        image_ft0 = FO.OpticsFFT(image0.values)
        for func in ft_func.keys():
            ft_image = (ft_func[func](image_ft0) / ft_scale[func])
            ft_image = ft_image - np.mean(ft_image)
            for ifilt in range(filter_set.sizes['filter_number']):
                # calculate projection of filter and particle FT
                image_svd[func].loc[{'hologram_number':im,'filter_number':ifilt}] = np.sum(filter_set.isel(filter_number=ifilt).values*ft_image)
        
        # report progress
        print(f"\r{im+1} of {ds['image'].sizes['hologram_number']} holograms completed",end='')
    ft_stop_time = datetime.datetime.now()
    ft_time = (ft_stop_time-ft_start_time).total_seconds()

    print(f"{ds['image'].shape[2]} samples Fourier Transformed in {ft_time} seconds")
    print(f"for {ft_time/ds['image'].shape[2]} seconds per hologram")

    # create data array of Fourier transformed data
    image_svd = xr.concat(list(image_svd.values()),pd.Index(ft_func.keys(),name='channels'))

    # assign FT data to the new xarray DataSet
    ds_ft = ds.assign(image_svd=image_svd,filter_set=filter_set)
    ds_ft = ds_ft.assign(particle_data=particle_data)

# set up encoding for non-floating point data type options
nckwargs = {}
if not 'float' in encoded_dtype:
    nckwargs['encoding'] = {'image_ft':{'dtype':encoded_dtype,
                            'scale_factor':(image_svd.max()-image_svd.min()).values/
                                (np.iinfo(encoded_dtype).max-np.iinfo(encoded_dtype).min),
                            'add_offset':image_svd.mean().values,
                            '_FillValue':np.iinfo(encoded_dtype).min}}

# write out the FT DataSet
nc_name = ds_base+"_svd_multipart"+"_".join(ft_func.keys())+"_"+"".join(part_components)+"_"+encoded_dtype+".nc"
print("saving data to")
print(ds_path+nc_name)
ds_ft.to_netcdf(ds_path+nc_name,**nckwargs)