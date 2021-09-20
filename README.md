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
* torchvision
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
    scikit-learn \
    torch \
    torchvision

```

* Ensure that CUDA kernel and CUDA toolkit are installed on your system, and the path and versions 

* Install the tensorflow binary for tensorflow 2. For more detailed installation instructions 
visit the [tensorflow website](https://www.tensorflow.org/install/gpu).
```bash
pip install tensorflow segmentation-models-pytorch
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

## Using holodec-ml
The repository contains python scripts and notebooks that allow users to train segmentation and classification models, and to perform inference with models on synthetic and real holograms.

### A. Configuration file
A user-supplied yml file is the basis for setting different parameters pertaining to datasets, resource usage, etc. For example, ```config/unet_propagation.yml``` contains the fields: seed, save_loc, data, transforms, model, optimizer, training, and inference.



```yaml
seed: 1000

save_loc: "/glade/work/schreck/repos/HOLO/clean/holodec-ml/results/optimized_noisy"

data:
    n_bins: 1000
    data_path: "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/synthetic_holograms_500particle_gamma_4872x3248_training.nc"
    raw_data: "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/real_holograms_CSET_RF07_20150719_203600-203700.nc"
    tile_size: 512
    step_size: 128
    marker_size: 10
    total_positive: 5
    total_negative: 5
    total_training: 50000
    output_path: "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/tiled_synthetic/"
    use_cached: True
    device: "cpu"
    cores: 0

transforms:
    training:
        RandomVerticalFlip:
            rate: 0.5
        RandomHorizontalFlip: 
            rate: 0.5
        Normalize:
            mode: '255'
        GaussianBlur:
            rate: 1.0
            kernel_size: 1
            sigma: 1.7902436563642792
        GaussianNoise:
            rate: 1.0
            noise: 0.011353967974406949
        AdjustBrightness:
            rate: 1.0
            brightness_factor: 1.7301154565769068
        ToTensor: True
    validation:
        Normalize:
            mode: '255'
        ToTensor: True
    inference:
        Normalize:
            mode: '255'       
            
model:
    name: "manet"
    encoder_name: "xception"
    encoder_weights: "imagenet"
    in_channels: 1
    classes: 1
    activation: "sigmoid"
    
optimizer:
    learning_rate: 0.0006145337230850794
    weight_decay: 0.0
    
trainer:
    epochs: 100
    train_batch_size: 16
    valid_batch_size: 16
    batches_per_epoch: 500
    stopping_patience: 4
    training_loss: "tyversky"
    validation_loss: "dice"
    
inference:
    mode: "mask"
    batch_size: 16
    n_nodes: 4
    gpus_per_node: 1
    threads_per_gpu: 2
    save_arrays: True
    save_probs: False
    probability_threshold: 0.5
    plot: False
    verbose: False
    data_set:
        path: "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/synthetic_holograms_500particle_gamma_4872x3248_test.nc"
        name: "synthetic"
#         path: "/glade/p/cisl/aiml/ai4ess_hackathon/holodec/real_holograms_CSET_RF07_20150719_200000-210000.nc"
#         name: "real"
        holograms:
            min: 0
            max: 10
```

There are two global fields for the user to set: 

* seed: Set a seed for reproducibility.
* save_loc: A file-path where all model training and inference results will be saved.

### B. Hologram data settings
There are two sets of data available: synthetically generated holograms, and real holograms from HOLODEC. The subfields within "data" in the configuration file allow one to point to them and describe how they will be preprocessed:

* n_bins: How many bins for partitioning (zmax - zmin).
* data_path: Path to synthetically generated data saved in xarray format.
* raw_path (optional): Path to real data obtained with HOLODEC.
* tile_size: The x and y size of subsampled images within full size images.
* step_size: The step measured in pixels between tiles.
* marker_size (deprecated): Used for generating Gaussian distributions around exact particle centers according to exp(-(r-r0)^2 / (2 marker_size)^2).
* total_positive: How many tiles to sample from a hologram where the particle is in-focus at some z.
* total_negative: How many tiles to sample from a hologram where the particle is out of at some z.  Approximately 1/2 of out-of-focus examples are selected from near in-focus particles, one z-bin away.
* total_training (optional): How many examples to save when caching a dataset.
* output_path (optional): Training, testing, and validation cached datasets will be saved using naming convention (tile_size)-(step_size)-(total_positive)-(total_negative)-(total_examples).
* use_cached: A flag to indicate whether to use a cached dataset.
* device: During training and inference phases, select the device where the wave-propagation preprocessing will occur.
* cores: The number of CPU cores to use in DataLoader instances during training.

### C. Image transformation settings

The "transforms" field allows one to specify which transformations are performed on the hologram tiles before being passed to a model. The three subfields allow the user flexibility on which transformations to use for different data splits: "training", "validation", and "inference". For other options, see ```holodecml/transforms.py```. 

The transformation "Normalize" has several options -- the default divides the raw image values by 255. The ToTensor transformation will allow the user to convert numpy tensors to torch tensors.

The rate parameter specified in several examples above indicates the probability at which the transformation will be applied. If the rate is set to unity in "GaussianBlur", "GaussianNoise", and "AdjustBrightness", the values for parameters "sigma", "noise", and "brightness_factor" will be sampled randomly between zero and the value specified, respectively. In other words, when rate = 1, sigma, noise, or the brightness factor indicate the maximum value that could be used. 

### D. Model settings

The available models are derived from the package segmentation-models-pytorch. The available settings shown above describe:

* name: The type of segmentation model to use. For example, unet.
* encoder_name: The type of model to use to build the encoder. For example, xception.
* encoder_weights: Name of the pretrained weights to use. For example, imagenet.
* in_channels: The number of input image channels. Default is 1.
* classes: The number of types of segmentation masks. Default is 1.
* activation: The output activation function to use. Default is sigmoid.

One may also pass any other option that is used by the base model class in segmentation-models-pytorch. See the documentation for more options, there are too many to list here.

### E. Optimizer settings

The optimizer is currently fixed as AdamW. The two available settings are:

* learning_rate: The initial learning rate to use. Default is 1e-4.
* weight_decay: The L2 penalty to apply. Default is zero.

### F. Model training settings 

The currently available settings for training segmentation models include:

* epochs: The maximum number of epochs to train a model
* train_batch_size: The batch size used when the model is in training mode
* valid_batch_size: The batch size used when the model is in evaluation mode
* batches_per_epoch: The number of batches taken to define one epoch
* stopping_patience: How many epochs to wait before stopping early, after a best model was saved.
* training_loss: The loss used during training.
* validation_loss: The loss used during validation.

A learning rate scheduler is also used during the training, which has patience set to one by default. 

### G. Inference settings

* mode: Either "mask" or "label" to indicate the type of model prediction. 
* batch_size: The batch size used when the model is in evaludation mode.
* n_nodes: How many nodes will be used to perform inference. Assumes the user will submit this many launch scripts.
* gpus_per_node: How many GPUs are available on the current node.
* threads_per_gpu: How many copies of the model to mount to each available GPU.
* save_arrays: Save truth and predicted segmentation arrays.
* save_probs: Save the predicted probabilities, where any p<0.5 is saved as zero, and otherwise three significant figures are saved as an integer.
* probability_threshold: The decision threshold.
* plot: Whether to plot the results at each z and save them to disk.
* verbose: Option allowing more verbose output.

The field "data_set" has the following settings:

* path: The dataset to be used for inference. 
* name: The name that should be assigned with this dataset. A directory with this name will be created under save_path and all inference data generated will be saved here.
* holograms: Has subfields min and max to refer to the range of holograms (by idx) to process. Default is min as 0 max as 10, as there are only 10 holograms in the "test" synthetic dataset.


## Training and Inference Workflow

### 1a. Caching a dataset for use in training models
One may generate and save to disk "training", "validation", and "testing" data splits (80/10/10) by running:

```python
python applications/data_generator.py config/unet_propagation.yml
```

which will save (tile image, binary label, segmentation mask) tuples to disk, where binary label is zero if the tile does not contain an in-focus particle, and one otherwise. The segmentation mask has the same dimension as the tile image and contains zeros everywhere except when there is a particle in-focus in the tile, where the pixels falling within the particles diameter are labeled 1.

The script loads a custom class that is based on the primitive in ```holodecml/propagation.py``` to perform wave propagation of holograms to some specified value of z. Then tiles are sampled to contain in- or out-of-focus particles. In this script, the out-of-focus examples are approximately sampled randomly half of the time if total_negative is greater than 1, while the other half is selected to be one bin away from an in-focus example along the z-axis. 

The user may speed up the wave propagation calculations by increasing the number of CPU cores (set the device to "cpu" if the number if larger than 4), or by using the GPU device (for example, "device:0"). Note that saving data consumes a large amount of disk-space. 

The examples generated for the three splits are saved using the tag-naming convention prefix-(tile_size)-(step_size)-(total_positive)-(total_negative)-(total_examples) where prefix will be "training", "validation", or "test."

### 1b. Preprocessing data on-the-fly
As caching datasets can be expensive, the user may also sample in- or out-of-focus particles at different z by setting the "use_cached" flag to false. A pytorch DataLoader object is instructed to load the UpsamplingReader class in ```holodecml/data.py```, which performs the wave-propagation calculation and the other preprocessing steps on full-sized holograms.

The DataLoader class allows the user to spawn multiple instances, as controlled by the "cores" setting, that each independently perform the preprocessing steps. One may experiment by running more than one preprocessing instance on the GPU.

### 2. Model training 
One may train a segmentation model by running:

```python
python applications/train.py config/unet_propagation.yml
```

which will perform several steps to train a model for a fixed number of epochs, and save the results using save_loc as the end-point. 

### 3. Inference

After a model has been trained it may be used to predict segmentation masks around particles at different values of z. To perform inference on a dataset, run the script:

```python
python applications/inference.py config/unet_propagation.yml
```

which will propagate the holgoram to each z-bin center and feed the derived tiles through the model. The script then performs an average over the tiles to recreate the original hologram image size. 

The script will save data as the user instructs for truth masks, predicted masks, and predicted probabilities. The "truth" mask used when real holograms are used is the result predicted by HoloSuite. 


### 4. Clustering and post-processing