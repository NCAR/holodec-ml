import torch
import logging
import torchvision
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import transforms, utils

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
# https://scikit-image.org/docs/dev/auto_examples/transform/plot_rescale.html


logger = logging.getLogger(__name__)


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        logger.info(f"Loaded Rescale transformation with output size {output_size}")

    def __call__(self, sample):
        image = sample['image']
        image_dim = image.shape
        
        if image_dim[-2] > self.output_size:
            frac = self.output_size / image_dim[-2]
        else:
            frac = image_dim[-2] / self.output_size
        
        image = image.reshape(image.shape[-2], image.shape[-1])
        image = rescale(image, frac, anti_aliasing=False)
        image = image.reshape(1, image.shape[-2], image.shape[-1])

        return {'image': image}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        logger.info(f"Loaded RandomCrop transformation")

    def __call__(self, sample):
        image= sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        return {'image': image}
      
        
class Standardize(object):
    """Standardize image"""
    def __init__(self):
        logger.info(f"Loaded Standardize transformation that rescales data so mean = 0, std = 1")
    
    def __call__(self, sample):
        image = sample['image']
        image = (image-np.mean(image)) / (np.std(image))
        return {'image': image}
    

class Normalize(object):
    """Normalize image"""
    def __init__(self):
        logger.info(f"Loaded Normalize transformation that normalizes data in the range 0 to 1")
    
    def __call__(self, sample):
        image = sample['image'].astype(np.float32)
        image -= image.min()
        image /= image.max()
        return {'image': image}
    
            
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, device = "cpu"):
        self.device = device
        logger.info(f"Loaded ToTensor transformation, putting tensors on device {device}")
    
    def __call__(self, sample):
        image = sample['image'].astype(np.float32)
        if len(image.shape) == 2:
            image = image.reshape(1, image.shape[0], image.shape[1])
        return {'image': torch.from_numpy(image)}