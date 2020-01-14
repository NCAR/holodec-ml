import numpy as np
import xarray as xr
import pandas as pd
import datetime

import FourierOpticsLib as FO


ds_path = "/scr/sci/mhayman/holodec/holodec-ml-data/"
# ds_name = "synthethic_holograms_v0.nc"
ds_name = "synthetic_holograms_v02.nc"  # 3 particle data
ds_base = ds_name.replace(".nc","")
particle_count = 3

# define the functions to be applied to the fourier transformed data
ft_func = {'real':np.real,'imag':np.imag,'amplitude':np.abs}
ft_scale = {'real':255,'imag':255,'amplitude':255}  # rescaling factors
encoded_dtype = "float"

print("Fourier Transforming: "+ds_name)
print("   path: "+ds_path)
print("   data type: "+encoded_dtype)
print("   output operations: ",end="")

with xr.open_dataset(ds_path+ds_name) as ds:
    # initialize particle training data
    particle_data = xr.DataArray(np.zeros((4,particle_count,ds['image'].sizes['hologram_number']),dtype=float),
                                dims=('particle_property','particle_number','hologram_number'),
                                coords={'particle_property':['x','y','z','d']})

    # initialize the image Fourier Transform channels
    image_ft = {}
    for func in ft_func.keys():
        print(func,end=", ")
        image_ft[func] = xr.DataArray(np.zeros(ds['image'].shape,dtype='float32'),
                            dims=ds['image'].dims) # ['ysize','xsize','hologram_number']

    print("")
    ft_start_time = datetime.datetime.now()
    # create a variable with for each Fourier Transform operation of each image
    for im in range(ds['image'].sizes['hologram_number']):
        # sort the particle data by hologram number
        particle_index = np.nonzero(ds['hid'].values==im+1)[0]  # find particles associated with this hologram
        particle_data.loc[{'hologram_number':im}]=ds[particle_data.coords['particle_property']].isel(particle=particle_index).to_array()

        image0 = ds['image'].isel(hologram_number=im) #.transpose('xsize','ysize')
        image_ft0 = FO.OpticsFFT(image0.values - np.mean(image0.values))

        for func in ft_func.keys():
            image_ft[func].loc[{'hologram_number':im}] = ft_func[func](image_ft0) / ft_scale[func]
            # image_ft[func][:,:,im] = ft_func[func](image_ft0) / ft_scale[func]
        print(f"\r{im+1} of {ds['image'].sizes['hologram_number']} holograms completed",end='')
    ft_stop_time = datetime.datetime.now()
    ft_time = (ft_stop_time-ft_start_time).total_seconds()

    print(f"{ds['image'].shape[2]} samples Fourier Transformed in {ft_time} seconds")
    print(f"for {ft_time/ds['image'].shape[2]} seconds per hologram")

    # create data array of Fourier transformed data
    image_ft = xr.concat(list(image_ft.values()),pd.Index(ft_func.keys(),name='channels'))

    # assign FT data to the new xarray DataSet
    ds_ft = ds.assign(image_ft=image_ft)
    ds_ft = ds_ft.assign(particle_data=particle_data)

# set up encoding for non-floating point data type options
nckwargs = {}
if not 'float' in encoded_dtype:
    nckwargs['encoding'] = {'image_ft':{'dtype':encoded_dtype,
                            'scale_factor':(image_ft.max()-image_ft.min()).values/
                                (np.iinfo(encoded_dtype).max-np.iinfo(encoded_dtype).min),
                            'add_offset':image_ft.mean().values,
                            '_FillValue':np.iinfo(encoded_dtype).min}}

# write out the FT DataSet
nc_name = ds_base+"_ft_multipart_"+"_".join(ft_func.keys())+"_"+encoded_dtype+".nc"
print("saving data to")
print(ds_path+nc_name)
ds_ft.to_netcdf(ds_path+nc_name,**nckwargs)