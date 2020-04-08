"""
Generate raw data of randomized pixels for
UNET image reconstruction
This is the raw data generator.
Preprocessing is typically required before
supplying to the UNET
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

Nsets = 5000  # number of training images to generate
wavelength = 355e-9
# zbins = 5 # number of histogram bins in z
binary_amplitude = True  # amplitude is binary
complevel = 9  # compression level

zmiss = 0  # value for when amplitude is zero
dspot = 5e-6  # resolvable spot size set by the aperture stop
Nparticles = 5  # number of particles per hologram
Prop_Scale = 4  # factor of times bigger propagation grid than detector
random_particle_count = True # randomize the number of particles up to Nparticles


# set the randomized space limits
param_lim = {'z':[0,2e-2],
             'amplitude':[0.2,1]}
            #  'Nrange':10}

# depth_array = np.linspace(param_lim['z'][0],param_lim['z'][1],param_lim['Nrange'])
# depth_array = np.array([param_lim['z'][1]])

# zedges = np.linspace(*param_lim['z'],zbins)

rspot = dspot/2

# set the size of the image
image_dim = {'x':256,
             'y':256,
             'pixel_width':3e-6}

print()
next_file = True
file_count = 1
while next_file:
    nc_name = f"UNET_image_{image_dim['x']}x{image_dim['y']}_{Nsets}count_{Nparticles}particles_v%02d.nc"%file_count
    if os.path.exists(ds_path+nc_name):
        file_count+=1
    else:
        next_file = False
        print("creating new raw file:")
        print(ds_path+nc_name)


# initialize simulation grid
grid = FO.Coordinate_Grid(((image_dim['y']*Prop_Scale,image_dim['x']*Prop_Scale,),
                           (image_dim['pixel_width'],image_dim['pixel_width'],))
                           ,inputType='ccd')
# reconstruction grid
grid2 = FO.Coordinate_Grid(((image_dim['y'],image_dim['x'],),
                           (image_dim['pixel_width'],image_dim['pixel_width'],))
                           ,inputType='ccd')
# initialize plane wave definition
E0 = FO.Efield(wavelength,grid,z=np.min(param_lim['z']))
E0.field = E0.field*np.sqrt(128.0)
OpticalTF = E0.grid.fr < 1/rspot


ds_attr = {}
ds_attr['xdim'] = image_dim['x']
ds_attr['ydim'] = image_dim['y']
ds_attr['pixel_width'] = image_dim['pixel_width']
ds_attr['wavelength'] = wavelength
ds_attr['zmax'] = max(param_lim['z'])
ds_attr['zmin'] = min(param_lim['z'])
ds_attr['z_invalid'] = zmiss
ds_attr['rspot'] = rspot
ds_attr['dspot'] = dspot
ds_attr['Nparticles'] = Nparticles
ds_attr['random_particle_count'] = np.int(random_particle_count)

for var in ds_attr:
    print(var+f":{ds_attr[var]}")


xsize = xr.DataArray(np.arange(image_dim['x']),
                                dims=('xsize'))
ysize = xr.DataArray(np.arange(image_dim['y']),
                                dims=('ysize'))
# depth is actually a combination of real and imaginary components along
# the z axis
# channel = xr.DataArray(np.arange(depth_array.size*2),
#                                 dims=('channel'))

# image = xr.DataArray(np.zeros((Nsets,image_dim['x'],image_dim['y'],channel.size),dtype=float),
#                 dims=('hologram_number','xsize','ysize','channel'),
#                 coords={'xsize':xsize,'ysize':ysize,'channel':channel})

# initial hologram image
imageh = xr.DataArray(np.zeros((Nsets,image_dim['x'],image_dim['y']),dtype=np.uint8),
                dims=('hologram_number','xsize','ysize'),
                coords={'xsize':xsize,'ysize':ysize})

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

    if random_particle_count:
        Nparticles = np.int(ds_attr['Nparticles']*np.random.rand()+1)
        
    

    zposition = sorted(np.random.rand(Nparticles)*(param_lim['z'][1]-param_lim['z'][0])+param_lim['z'][0])
    E1 = E0.copy()
    adata = np.ones((image_dim['x']*Prop_Scale,image_dim['y']*Prop_Scale))
    zdata = np.ones((image_dim['x']*Prop_Scale,image_dim['y']*Prop_Scale))*zmiss
    
    for npart in range(Nparticles):
        # create the particle position
        pix_x = np.int(np.random.rand()*image_dim['x'])
        pix_y = np.int(np.random.rand()*image_dim['y'])

        # set up the particle amplitudegrid
        adatap = np.zeros((image_dim['x']*Prop_Scale,image_dim['y']*Prop_Scale))
        # adata0 = np.zeros((image_dim['x'],image_dim['y']))
        
        # create the random particle
        p_grow = 0.8+0.19*np.random.rand() # probability of growth in particle definition
        p_decay = 0.8+0.19*np.random.rand() # decay rate of growth probability

        ml.next_pt((pix_x+grid.Nx*(Prop_Scale-1)//(Prop_Scale*2), \
            pix_y+grid.Nx*(Prop_Scale-1)//(Prop_Scale*2)), \
                adatap,p_grow,decay=p_decay)

        if not binary_amplitude:
            # let amplitude be any number between 0 and 1
            # otherwise the object is binary
            ipix = np.nonzero(adatap)
            adatap[ipix] = np.random.rand(len(ipix[0]))*(max(param_lim['amplitude'])-min(param_lim['amplitude']))+min(param_lim['amplitude'])
        
        # create the z position array for training
        ipix = np.nonzero(adatap)
        zdata[ipix] = zposition[npart]

        # propagate the electric field to the new particle
        # and apply the particle amplitude mask
        E1.propagate_to(zposition[npart])
        E1.field *= (1-adatap)

        adata *= (1-adatap)

    E1.propagate_to(np.max(param_lim['z']))
    E1.spatial_filter(OpticalTF)

    image0 = np.abs(E1.field[grid.Nx*(Prop_Scale-1)//(Prop_Scale*2):-grid.Nx*(Prop_Scale-1)//(Prop_Scale*2), \
        grid.Ny*(Prop_Scale-1)//(Prop_Scale*2):-grid.Ny*(Prop_Scale-1)//(Prop_Scale*2)])**2

    adata0 = 1-adata[grid.Nx*(Prop_Scale-1)//(Prop_Scale*2):-grid.Nx*(Prop_Scale-1)//(Prop_Scale*2), \
        grid.Ny*(Prop_Scale-1)//(Prop_Scale*2):-grid.Ny*(Prop_Scale-1)//(Prop_Scale*2)]
    zdata0 = zdata[grid.Nx*(Prop_Scale-1)//(Prop_Scale*2):-grid.Nx*(Prop_Scale-1)//(Prop_Scale*2), \
        grid.Ny*(Prop_Scale-1)//(Prop_Scale*2):-grid.Ny*(Prop_Scale-1)//(Prop_Scale*2)]
    
    imageft0 = FO.OpticsFFT(image0)

    labels.loc[{'hologram_number':iholo,'type':'z'}] = zdata0
    labels.loc[{'hologram_number':iholo,'type':'amplitude'}] = adata0
    imageh.loc[{'hologram_number':iholo}] = image0.astype(np.uint8)
    # image_ft.loc[{'hologram_number':iholo,'channel':'real'}] = np.real(imageft0)
    # image_ft.loc[{'hologram_number':iholo,'channel':'imag'}] = np.imag(imageft0)

    # report progress
    print(f"\r{iholo+1} of {Nsets} holograms completed",end='')


ds = xr.Dataset({'xsize':xsize,'ysize':ysize,'image':imageh,'labels':labels,},
                attrs=ds_attr)

# ds = xr.Dataset({'xsize':xsize,'ysize':ysize,'image':image,'labels':labels,'channel':channel},
#                 attrs=ds_attr)

print("saving data with compression level %d to"%complevel)
print(ds_path+nc_name)

nccomp = dict(zlib=True, complevel=complevel)
ncencoding = {var: nccomp for var in ds.data_vars}
ds.to_netcdf(ds_path+nc_name, encoding=ncencoding,format='netCDF4',engine='netcdf4',shuffle=True)

print('save complete')
