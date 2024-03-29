import numpy as np
import xarray as xr
import pandas as pd
import datetime

import FourierOpticsLib as FO


ds_path = "/scr/sci/mhayman/holodec/holodec-ml-data/"
ds_name = "synthethic_holograms_v0.nc"
ds_base = ds_name.replace(".nc","")

# define the functions to be applied to the fourier transformed data
ft_func = {'real':np.real,'imag':np.imag,'amplitude':np.abs}
ft_scale = {'real':255,'imag':255,'amplitude':255}  # rescaling factors
encoded_dtype = "int16"

print("Fourier Transforming: "+ds_name)
print("   path: "+ds_path)
print("   data type: "+encoded_dtype)
print("   output operations: ",end="")

with xr.open_dataset(ds_path+ds_name) as ds:
    # initialize the image Fourier Transform channels
    image_ft = {}
    for func in ft_func.keys():
        print(func,end=", ")
        image_ft[func] = xr.DataArray(np.zeros(ds['image'].shape,dtype='float32'),
                            dims=['ysize','xsize','hologram_number'])

    print("")
    ft_start_time = datetime.datetime.now()
    # create a variable with for each Fourier Transform operation of each image
    for im in range(ds['image'].shape[2]):
        image_ft0 = FO.OpticsFFT(ds['image'][:,:,im]-np.mean(ds['image'][:,:,im]))
        for func in ft_func.keys():
            image_ft[func][:,:,im] = ft_func[func](image_ft0) / ft_scale[func]
    ft_stop_time = datetime.datetime.now()
    ft_time = (ft_stop_time-ft_start_time).total_seconds()

    print(f"{ds['image'].shape[2]} samples Fourier Transformed in {ft_time} seconds")
    print(f"for {ft_time/ds['image'].shape[2]} seconds per hologram")

    # create data array of Fourier transformed data
    image_ft = xr.concat(list(image_ft.values()),pd.Index(ft_func.keys(),name='channels'))

    # assign FT data to the new xarray DataSet
    ds_ft = ds.assign(image_ft=image_ft)

# set up encoding for non-floating point data type options
nckwargs = {}
if not 'float' in encoded_dtype:
    nckwargs['encoding'] = {'image_ft':{'dtype':encoded_dtype,
                            'scale_factor':(image_ft.max()-image_ft.min()).values/
                                (np.iinfo(encoded_dtype).max-np.iinfo(encoded_dtype).min),
                            'add_offset':image_ft.mean().values,
                            '_FillValue':np.iinfo(encoded_dtype).min}}

# write out the FT DataSet
ds_ft.to_netcdf(ds_path+ds_base+"_ft_ac_"+"_".join(ft_func.keys())+"_"+encoded_dtype+".nc",**nckwargs)