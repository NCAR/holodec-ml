import torch
import random
import logging
import torchvision
import numpy as np
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
# https://scikit-image.org/docs/dev/auto_examples/transform/plot_rescale.html


logger = logging.getLogger(__name__)


def LoadTransformations(transform_config: str):
    tforms = []
    if "RandomVerticalFlip" in transform_config:
        rate = transform_config["RandomVerticalFlip"]["rate"]
        if rate > 0.0:
            tforms.append(RandVerticalFlip(rate))
    if "RandomHorizontalFlip" in transform_config:
        rate = transform_config["RandomVerticalFlip"]["rate"]
        if rate > 0.0:
            tforms.append(RandHorizontalFlip(rate))
    if "Rescale" in transform_config:
        rescale = transform_config["Rescale"]
        tforms.append(Rescale(rescale))
    if "Normalize" in transform_config:
        mode = transform_config["Normalize"]["mode"]
        tforms.append(Normalize(mode))
    if "GaussianNoise" in transform_config:
        rate = transform_config["GaussianNoise"]["rate"]
        noise = transform_config["GaussianNoise"]["noise"]
        if rate > 0.0:
            tforms.append(GaussianNoise(rate,noise))
    if "ToTensor" in transform_config:
        tforms.append(ToTensor())
    if "AdjustBrightness" in transform_config:
        rate = transform_config["AdjustBrightness"]["rate"]
        brightness = transform_config["AdjustBrightness"]["brightness_factor"]
        if rate > 0.0:
            tforms.append(AdjustBrightness(rate, brightness))
    if "GaussianBlur" in transform_config:
        rate = transform_config["GaussianBlur"]["rate"]
        k_sz = transform_config["GaussianBlur"]["kernel_size"]
        sigma = transform_config["GaussianBlur"]["sigma"]
        if rate > 0.0:
            tforms.append(GaussianBlur(rate, k_sz, brightness))
    if "RandomCrop" in transform_config:
        tforms.append(RandomCrop())
    if "Standardize" in transform_config:
        tforms.append(Standardize())
    #transform = transforms.Compose(tforms)
    return tforms


class RandVerticalFlip(object):

    def __init__(self, rate):
        logger.info(
            f"Loaded RandomVerticalFlip transformation with probability {rate}")
        self.rate = rate

    def __call__(self, sample):
        image = sample['image']
        flipped = False
        if random.random() < self.rate:
            image = np.flip(image, axis=2)
            flipped = True
        sample["image"] = image
        sample["vertical_flip"] = flipped
        return sample


class RandHorizontalFlip(object):

    def __init__(self, rate):
        logger.info(
            f"Loaded RandomHorizontalFlip transformation with probability {rate}")
        self.rate = rate

    def __call__(self, sample):
        image = sample['image']
        flipped = False
        if random.random() < self.rate:
            image = np.flip(image, axis=1)
            flipped = True
        sample["image"] = image
        sample["horizontal_flip"] = flipped
        return sample


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
        logger.info(
            f"Loaded Rescale transformation with output size {output_size}")

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

        sample["image"] = image
        return sample


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
        image = sample['image']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        
        sample["image"] = image
        return sample


class Standardize(object):
    """Standardize image"""

    def __init__(self):
        logger.info(
            f"Loaded Standardize transformation that rescales data so mean = 0, std = 1")

    def __call__(self, sample):
        image = sample['image']
        image = (image-image.mean()) / (image.std())
        sample["image"] = image
        return sample


class Normalize(object):
    """Normalize image"""

    def __init__(self, mode="norm"):
        if mode == "norm":
            logger.info(
                f"Loaded Normalize transformation that normalizes data in the range 0 to 1")
        if mode == "sym":
            logger.info(
                f"Loaded Normalize transformation that normalizes data in the range -1 to 1")
        if mode == "255":
            logger.info(
                f"Loaded Normalize transformation that normalizes data color channel by dividing by 255.0 and phase pi")
        self.mode = mode

    def __call__(self, sample):
        
        image = sample['image'] #.astype(np.float32)
        
        if self.mode == "norm":
            image -= image.min()
            image /= image.max()

        if self.mode == "sym":
            image = -1 + 2.0*(image - image.min())/(image.max() - image.min())
            
        if self.mode == "255":
            #image /= 255.0
            image[0] /= 255.0
            if image.shape[0] > 1:
                image[1] = (1.0 + image[1] / np.pi) / 2.0
        
        sample["image"] = image
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        logger.info(
            f"Loaded ToTensor transformation")

    def __call__(self, sample):
        image = sample['image'].astype(np.float32)
        if len(image.shape) == 2:
            image = image.reshape(1, image.shape[0], image.shape[1])
        sample["image"] = torch.from_numpy(image).float()
        return sample
    
    
class AdjustBrightness(object):
    
    def __init__(self, rate, brightness):
        logger.info(
            f"Loaded AdjustBrightness transformation")
        self.rate = rate
        self.brightness = brightness

    def __call__(self, sample):
        if self.rate >= 1.0:
            image = sample['image']
            brightness = random.uniform(0.0, self.brightness)
            image = torchvision.transforms.functional.adjust_brightness(
                image, brightness
            )
            sample["image"] = image
        elif random.random() < self.rate:
            image = sample['image']
            image = torchvision.transforms.functional.adjust_brightness(
                image, self.brightness
            )
            sample["image"] = image
        return sample
    
class GaussianBlur(object):
    
    def __init__(self, rate, kernel_size, sigma):
        logger.info(
            f"Loaded GaussianBlur transformation")
        self.rate = rate
        self.kernel_size = int(kernel_size)
        self.sigma = sigma

    def __call__(self, sample):
        if self.rate >= 1.0:
            image = sample['image']
            sigma = random.uniform(0.0, self.sigma)
            image = torchvision.transforms.functional.gaussian_blur(
                image, 
                kernel_size = self.kernel_size, 
                sigma = sigma
            )
            sample["image"] = image
        elif random.random() < self.rate:
            image = sample['image']
            image = torchvision.transforms.functional.gaussian_blur(
                image, 
                kernel_size = self.kernel_size, 
                sigma = self.sigma
            )
            sample["image"] = image
        return sample
    

class GaussianNoise(object):
    
    def __init__(self, rate, noise):
        logger.info(
            f"Loaded GaussianNoise transformation")
        self.rate = rate
        self.noise = noise

    def __call__(self, sample):
        if self.rate >= 1.0:
            image = sample['image']
            noise = random.uniform(0.0, self.noise)
            noise = np.random.normal(0, noise, image.shape)
            image += noise
            sample["image"] = image
        elif random.random() < self.rate:
            image = sample['image']
            noise = np.random.normal(0, self.noise, image.shape)
            image += noise
            sample["image"] = image
        return sample