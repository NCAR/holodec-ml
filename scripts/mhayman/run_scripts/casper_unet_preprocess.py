"""
Run Script for Creating a Preprocessed Training Dataset
On Casper

TODO: Add Fourier Optics libraries to python path
"""

software_path = "/glade/u/home/mhayman/Python/holodec-ml/scripts/mhayman/"

ds_path = "/glade/scratch/mhayman/holodec/holodec-ml-data/UNET/"
ds_file = "UNET_image_256x256_5000count_5particles_v02.nc"

rescale = 255

# specify number of layers to reconstruct
params = {'zplanes':2,
          'preprocess_type':'multi-plane reconstruction',
          'raw_file':ds_file,
          'complevel':9}

exec_file = software_path+"PreProcess/PreProcess_MultiPlane_UNET.py"
exec(compile(open(exec_file,'rb').read(),exec_file,'exec'))