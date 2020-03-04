"""
Generate randomized pixel data for 
U-net image reconstruction
Treat the problem as a classification problem
"""
import sys
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

ds_path = "/scr/sci/mhayman/holodec/holodec-ml-data/"

Nsets = 5000  # number of training images to generate
wavelength = 660e-9
zbins = 5 # number of histogram bins in z
binary_amplitude = True  # amplitude is binary

zmiss = 0  # value for when amplitude is zero
rspot = 1e-6  # resolvable spot size set by the aperture stop
Nparticles = 1  # number of particles per hologram



# set the randomized space limits
param_lim = {'z':[0,1e-2],
             'amplitude':[0.2,1],
             'Nrange':4}

depth_array = np.linspace(param_lim['z'][0],param_lim['z'][1],param_lim['Nrange'])

zedges = np.linspace(*param_lim['z'],zbins)

# set the size of the image
image_dim = {'x':64,
             'y':64,
             'pixel_width':10e-6}

nc_name = f"random_image_multiplane_data_{image_dim['x']}x{image_dim['x']}_{Nsets}count_{Nparticles}particles.nc"

# initialize simulation grid
grid = FO.Coordinate_Grid(((image_dim['y']*2,image_dim['x']*2,),
                           (image_dim['pixel_width'],image_dim['pixel_width'],))
                           ,inputType='ccd')
# reconstruction grid
grid2 = FO.Coordinate_Grid(((image_dim['y'],image_dim['x'],),
                           (image_dim['pixel_width'],image_dim['pixel_width'],))
                           ,inputType='ccd')
# initialize plane wave definition
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
ds_attr['Nparticles'] = Nparticles

for var in ds_attr:
    print(var+f":{ds_attr[var]}")


xsize = xr.DataArray(np.arange(image_dim['x']),
                                dims=('xsize'))
ysize = xr.DataArray(np.arange(image_dim['y']),
                                dims=('ysize'))
# depth is actually a combination of real and imaginary components along
# the z axis
channel = xr.DataArray(np.arange(depth_array.size*2),
                                dims=('channel'))

image = xr.DataArray(np.zeros((Nsets,image_dim['x'],image_dim['y'],channel.size),dtype=float),
                dims=('hologram_number','xsize','ysize','channel'),
                coords={'xsize':xsize,'ysize':ysize,'channel':channel})

# image_ft = xr.DataArray(np.zeros((Nsets,image_dim['x'],image_dim['y'],2),dtype=float),
#                 dims=('hologram_number','xsize','ysize','channel'),
#                 coords={'xsize':xsize,'ysize':ysize,'channel':['real','imag']})

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
    

    zposition = sorted(np.random.rand(Nparticles)*(param_lim['z'][1]-param_lim['z'][0])+param_lim['z'][0])
    E1 = E0.copy()
    adata = np.ones((image_dim['x']*2,image_dim['y']*2))
    zdata = np.ones((image_dim['x']*2,image_dim['y']*2))*zmiss
    
    for npart in range(Nparticles):
        # create the particle position
        pix_x = np.int(np.random.rand()*image_dim['x'])
        pix_y = np.int(np.random.rand()*image_dim['y'])

        # set up the particle amplitudegrid
        adatap = np.zeros((image_dim['x']*2,image_dim['y']*2))
        # adata0 = np.zeros((image_dim['x'],image_dim['y']))
        
        # create the random particle
        ml.next_pt((pix_x+grid.Nx//4,pix_y+grid.Nx//4),adatap,0.8,decay=0.9)

        if not binary_amplitude:
            # let amplitude be any number between 0 and 1
            # otherwise the object is binary
            ipix = np.nonzero(adatap)
            adatap[ipix] = np.random.rand(len(ipix[0]))*(max(param_lim['amplitude'])-min(param_lim['amplitude']))+min(param_lim['amplitude'])

        # adata[grid.Nx//4:-grid.Nx//4,grid.Ny//4:-grid.Ny//4] = adata0

        # zdata = np.ones((image_dim['x'],image_dim['y']))*zmiss
        
        # create the z position array for training
        ipix = np.nonzero(adatap)
        zdata[ipix] = zposition[npart]

        # propagate the electric field to the new particle
        # and apply the particle amplitude mask
        E1.propagate_to(zposition[npart])
        E1.field *= (1-adatap)
        # E1.spatial_filter(OpticalTF)

        adata *= (1-adatap)

    E1.propagate_to(np.max(param_lim['z']))
    E1.spatial_filter(OpticalTF)

    image0 = np.abs(E1.field[grid.Nx//4:-grid.Nx//4,grid.Ny//4:-grid.Ny//4])**2

    adata0 = 1-adata[grid.Nx//4:-grid.Nx//4,grid.Ny//4:-grid.Ny//4]
    zdata0 = zdata[grid.Nx//4:-grid.Nx//4,grid.Ny//4:-grid.Ny//4]
    
    # initialize the reconstruction
    E2 = FO.Efield(wavelength,grid2,z=E1.z,fielddef=image0)
    for idepth,depthr in enumerate(depth_array):
        E2.propagate_to(depthr)
        image.loc[{'hologram_number':iholo,'channel':2*idepth}] = np.real(E2.field)
        image.loc[{'hologram_number':iholo,'channel':2*idepth+1}] = np.imag(E2.field)

    # imageft0 = FO.OpticsFFT(image0)

    labels.loc[{'hologram_number':iholo,'type':'z'}] = zdata0
    labels.loc[{'hologram_number':iholo,'type':'amplitude'}] = adata0
    # image.loc[{'hologram_number':iholo}] = image0.copy()
    # image_ft.loc[{'hologram_number':iholo,'channel':'real'}] = np.real(imageft0)
    # image_ft.loc[{'hologram_number':iholo,'channel':'imag'}] = np.imag(imageft0)

    # report progress
    print(f"\r{iholo+1} of {Nsets} holograms completed",end='')


ds = xr.Dataset({'xsize':xsize,'ysize':ysize,'image':image,'labels':labels,'channel':channel},
                attrs=ds_attr)

print("saving data to")
print(ds_path+nc_name)
ds.to_netcdf(ds_path+nc_name)
