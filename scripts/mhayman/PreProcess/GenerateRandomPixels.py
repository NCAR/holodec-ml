"""
Generate randomized pixel data for 
U-net image reconstruction
"""

import numpy as np
import xarray as xr
import datetime
import copy

import FourierOpticsLib as FO

ds_path = "/scr/sci/mhayman/holodec/holodec-ml-data/"

Nsets = 5000  # number of training images to generate
wavelength = 355e-9
amplitude_th = 0.01  # mean fraction of zero pixels
binary_amplitude = False  # amplitude is binary
zmiss = 0  # value for when amplitude is zero
rspot = 10e-6  # resolvable spot size set by the aperture stop



# set the randomized space limits
param_lim = {'z':[-1e-3,0],
             'amplitude':[0,1]}

# set the size of the image
image_dim = {'x':64,
             'y':64,
             'pixel_width':3e-6}

nc_name = f"image_data_{image_dim['x']}x{image_dim['x']}_{Nsets}count.nc"

# initialize plane wave definition
grid = FO.Coordinate_Grid(((image_dim['y']*2,image_dim['x']*2,),
                           (image_dim['pixel_width'],image_dim['pixel_width'],))
                           ,inputType='ccd')
E0 = FO.Efield(wavelength,grid,z=np.min(param_lim['z']))
OpticalTF = E0.grid.fr < 1/rspot


ds_attr = {}
ds_attr['xdim'] = image_dim['x']
ds_attr['xdim'] = image_dim['y']
ds_attr['pixel_width'] = image_dim['pixel_width']
ds_attr['wavelength'] = wavelength
ds_attr['zmax'] = max(param_lim['z'])
ds_attr['zmin'] = min(param_lim['z'])
ds_attr['z_invalid'] = zmiss
ds_attr['rspot'] = rspot

for var in ds_attr:
    print(var+f":{ds_attr[var]}")


xsize = xr.DataArray(np.arange(image_dim['x']),
                                dims=('xsize'))
ysize = xr.DataArray(np.arange(image_dim['y']),
                                dims=('ysize'))
image = xr.DataArray(np.zeros((Nsets,image_dim['x'],image_dim['y']),dtype=float),
                dims=('hologram_number','xsize','ysize'),
                coords={'xsize':xsize,'ysize':ysize})

image_ft = xr.DataArray(np.zeros((Nsets,image_dim['x'],image_dim['y'],2),dtype=float),
                dims=('hologram_number','xsize','ysize','channel'),
                coords={'xsize':xsize,'ysize':ysize,'channel':['real','imag']})

labels = xr.DataArray(np.zeros((Nsets,image_dim['x'],image_dim['y'],2),dtype=float),
                dims=('hologram_number','xsize','ysize','type'),
                coords={'xsize':xsize,'ysize':ysize,'type':['z','amplitude']},
                attrs={'description':'pixel properties for training'})

# labels_a = xr.DataArray(np.zeros((Nsets,image_dim['x'],image_dim['y']),dtype=float),
#                 dims=('hologram_number','xsize','ysize'),
#                 coords={'xsize':xsize,'ysize':ysize},
#                 attrs={'description':'transmission amplitude of the pixel'})


"""
Generate pixel input data
loop
"""
for iholo in range(Nsets):
    if binary_amplitude:
        # force amplitude data to be binary
        adata = (np.random.rand(image_dim['x']*2,image_dim['y']*2) < amplitude_th).astype(float)  
    else:
        # let amplitude be any number between 0 and 1
        adata = np.maximum(np.random.rand(image_dim['x']*2,image_dim['y']*2)-(1-amplitude_th),0.0)/amplitude_th

    # eliminate points outside of the 
    # actual image
    adata[:grid.Nx//4,:] = 0
    adata[-grid.Nx//4:,:] = 0
    adata[:,:grid.Ny//4] = 0
    adata[:,-grid.Ny//4:] = 0

    zdata = np.random.rand(image_dim['x']*2,image_dim['y']*2)*(param_lim['z'][1]-param_lim['z'][0])+param_lim['z'][0]
    zdata[adata<=0] = zmiss  # null zdata where there is no amplitude term

    """
    Two choices - include multiple scattering by numerically
    propagating the wave in order of z for each pixel
    OR
    analytically calculate the resulting hologram considering
    only 

    """

    # sort pixels by z position
    ipix = np.argsort(zdata.flatten())
    ipix = np.delete(ipix,np.nonzero(adata.flatten()==1))
    ix,iy = np.meshgrid(np.arange(2*image_dim['x']),np.arange(2*image_dim['y']))
    ix = ix.ravel()
    iy = iy.ravel()

    E1 = E0.copy()

    for i in ipix:
        if adata[ix[i],iy[i]] > 0:
            E1.propagate_to(zdata[ix[i],iy[i]])
            # E1.propagate_Fresnel(zdata[ix[i],iy[i]]-E1.z)  # use Fresnel propagation
            E1.field[ix[i],iy[i]] *= (1-adata[ix[i],iy[i]])
            # E1.spatial_filter(OpticalTF)

    image0 = np.abs(E1.field[grid.Nx//4:-grid.Nx//4,grid.Ny//4:-grid.Ny//4])**2
    imageft0 = FO.OpticsFFT(image0)

    labels.loc[{'hologram_number':iholo,'type':'z'}] = zdata[grid.Nx//4:-grid.Nx//4,grid.Ny//4:-grid.Ny//4]
    labels.loc[{'hologram_number':iholo,'type':'amplitude'}] = adata[grid.Nx//4:-grid.Nx//4,grid.Ny//4:-grid.Ny//4]
    image.loc[{'hologram_number':iholo}] = image0.copy()
    image_ft.loc[{'hologram_number':iholo,'channel':'real'}] = np.real(imageft0)
    image_ft.loc[{'hologram_number':iholo,'channel':'imag'}] = np.imag(imageft0)

    # report progress
    print(f"\r{iholo+1} of {Nsets} holograms completed",end='')


ds = xr.Dataset({'xsize':xsize,'ysize':ysize,'image':image,'labels':labels,'image_ft':image_ft},
                attrs=ds_attr)

print("saving data to")
print(ds_path+nc_name)
ds.to_netcdf(ds_path+nc_name)
