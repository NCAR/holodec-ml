## Contributers 
* David John Gagne
* Aaron Bansemer
* Matthew Hayman
* John Shreck 
* Gabrielle Gantos
* Gunther Wallach

## Requirments
The code is designed to run on Python 3.7. It reqires the following Pythong libraries:
* os
* xarray
* numpy 
* pandas
* typing
* tensorflow>=2.0.0
* netcdf4

## Setup from Scratch

* Install Python 3.7 on your machine. I recommend the Miniconda Python installer available
[here](https://docs.conda.io/en/latest/miniconda.html).

* Create a conda environment for holodec:  `conda create -n holodec python=3.7`

* Activate the environment on your machine:  
`source activate goes16`

* Install the Python libraries through conda:

```bash
conda install -c conda-forge --yes \
    pip \
    os \
    xarray \
    numpy \
    pandas \
    typing \
    netcdf4 \
```

* Ensure that CUDA kernel and CUDA toolkit are installed on your system, and the path and versions 

* Install the tensorflow binary for tensorflow 2. For more detailed installation instructions 
visit the [tensorflow website](https://www.tensorflow.org/install/gpu).
```bash
pip install tensorflow
```

* Clone the holodec library into your directory
```bash
cd ~
git clone https://github.com/NCAR/holodec-ml.git
cd holodec-ml
```

* Install the holodec library
```bash
pip install .
```




