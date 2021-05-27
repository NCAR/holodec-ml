## Contributers 
* David John Gagne
* Aaron Bansemer
* Matthew Hayman
* John Schreck 
* Gabrielle Gantos
* Gunther Wallach

## Requirments
The code is designed to run on Python 3.7. It reqires the following external Python libraries:
* xarray
* numpy<1.19
* scipy
* pandas
* scikit-learn
* tensorflow>=2.0.0
* torch
* netcdf4
* tqdm
* Pillow

## Setup from Scratch

* Install Python 3.7 on your machine. I recommend the Miniconda Python installer available
[here](https://docs.conda.io/en/latest/miniconda.html).

* Create a conda environment for holodec:  `conda create -n holodec python=3.7`

* Activate the environment on your machine:  
`source activate holodec`

* Install the Python libraries through conda:

```bash
conda install -c conda-forge --yes \
    pip \
    tqdm \
    xarray \
    "numpy<1.19" \
    pandas \
    netcdf4 \
    scikit-learn

```

* Ensure that CUDA kernel and CUDA toolkit are installed on your system, and the path and versions 

* Install the tensorflow binary for tensorflow 2. For more detailed installation instructions 
visit the [tensorflow website](https://www.tensorflow.org/install/gpu).
```bash
pip install tensorflow
```
* Install Pytorch using the instructions on the [PyTorch website](https://pytorch.org/). 

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

