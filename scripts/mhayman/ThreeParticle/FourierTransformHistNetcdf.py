"""

"""

import numpy as np
import xarray as xr
import pandas as pd
import datetime
import FourierOpticsLib as FO

ds_path = "/scr/sci/mhayman/holodec/holodec-ml-data/"
ds_name = "synthetic_holograms_v02.nc"
ds_base = ds_name.replace(".nc","")

# define the functions to be applied to the fourier transformed data
# this sets the out channels
ft_func = {'real':np.real,'imag':np.imag}
encoded_dtype = "float32"

# define the data products desired and the number of histogram bins
# this defines the output histogram dimensions
hist_bin_count = {"d":25}

# general definitions for rescaling factors
ft_scale = {'real':255,'imag':255,'amplitude':255,'phase':2*np.pi}  # rescaling factors

print("Fourier Transforming: "+ds_name)
print("   path: "+ds_path)
print("   data type: "+encoded_dtype)
print("   output operations: ",end="")

with xr.open_dataset(ds_path+ds_name) as ds:
    # initialize the particle property histogram bins
    hist_bin_centers = {}
    hist_bin_edges = {}
    histogram_shape = []
    histogram_size = 1
    hist_dims = ['hologram_number']
    hist_coords = [ds['hologram_number']]
    for param in hist_bin_count.keys():
        hist_bin_edges[param+'_bin_edges'] = np.linspace(ds[param].min(),ds[param].max(),
                            hist_bin_count[param])
        hist_bin_centers[param+'_bin_centers'] = 0.5*(hist_bin_edges[param+'_bin_edges'] [:-1]+
                            hist_bin_edges[param+'_bin_edges'][1:])
        histogram_shape+=[hist_bin_count[param]-1]
        histogram_size *= (hist_bin_count[param]-1)


    # initialize the image Fourier Transform channels
    image_ft = {}
    for func in ft_func.keys():
        image_ft[func] = xr.DataArray(np.zeros(ds['image'].shape,dtype='float32'),
                            dims=['hologram_number','xsize','ysize'])
        print(func,end=", ")
    print()

    print("   histogram bins: ")
    print("      "+hist_bin_count.__str__())

    # initialize the histograms
    particle_histogram = xr.DataArray(np.zeros([ds['image'].shape[0]]+histogram_shape),
                                coords=[ds['hologram_number']]+list(hist_bin_centers.values()),
                                dims=['hologram_number']+list(hist_bin_centers.keys()))


    ds_ft = ds.assign(**hist_bin_edges)     # create a new data set with histogram bin edges
    ds_ft = ds_ft.assign_coords(**hist_bin_centers) # create dimension for histogram bin centers
    ds_ft = ds_ft.assign(particle_histogram=particle_histogram) # add the histogram to the data set
    


    # store the Fourier Transform and particle size histogram for each hologram
    print("Performing Fourier Transform")
    ft_start_time = datetime.datetime.now()
    for im in ds_ft['hologram_number'].values:
        # find the particles in this hologram
        # hologram indexing is base 1
        particle_index = np.nonzero(ds_ft['hid'].values==im+1)[0]  
        
        # make a histogram of particles and store it in the data set
        hist0 = np.histogramdd(ds_ft[list(hist_bin_count.keys())].to_array()[:,particle_index].T,
                    bins=list(hist_bin_edges.values()))
        ds_ft['particle_histogram'][im,...] = hist0[0]
        
        # FT the image and store the desired operations
        image0 = ds['image'].sel(hologram_number=im)  # select the hologram image
        image_ft0 = FO.OpticsFFT(image0-np.mean(image0))  # FFT the image
        # perform requested operations for storage
        for func in ft_func.keys():
            image_ft[func][im,:,:] = ft_func[func](image_ft0) / ft_scale[func]
ft_stop_time = datetime.datetime.now()

image_ft = xr.concat(list(image_ft.values()),pd.Index(ft_func.keys(),name='channel'))
image_ft = image_ft.transpose("hologram_number", "xsize", 'ysize',"channel")  # fix the axis order
ds_ft = ds_ft.assign(image_ft=image_ft)

ft_time = (ft_stop_time-ft_start_time).total_seconds()
print(f"{ds['image'].shape[2]} samples Fourier Transformed in {ft_time} seconds")
print(f"for {ft_time/ds['image'].shape[2]} seconds per hologram")

# set up encoding for non-floating point data type options
nckwargs = {}
if not 'float' in encoded_dtype:
    nckwargs['encoding'] = {'image_ft':{'dtype':encoded_dtype,
                            'scale_factor':(image_ft.max()-image_ft.min()).values/
                                (np.iinfo(encoded_dtype).max-np.iinfo(encoded_dtype).min),
                            'add_offset':image_ft.mean().values,
                            '_FillValue':np.iinfo(encoded_dtype).min}}

# write out the FT DataSet
print("Writing to netcdf")
ds_ft.to_netcdf(ds_path+ds_base+"_ft_ac_"+
    "".join(hist_bin_count.keys())+"_"+
    "_".join(ft_func.keys())+"_"+
    encoded_dtype+".nc",**nckwargs)