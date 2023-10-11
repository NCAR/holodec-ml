import torch
import random
import logging
import torchvision
import numpy as np

from torch import Tensor
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, __version__ as PILLOW_VERSION

from torchvision import datasets
from torchvision import transforms
#from torchvision import functional

from torchvision.utils import save_image
from torchvision import utils

from typing import Tuple, List, Optional

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
from scipy.ndimage.filters import gaussian_filter


# https://scikit-image.org/docs/dev/auto_examples/transform/plot_rescale.html
# https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py


logger = logging.getLogger(__name__)


# "RandomPosterize",
# "RandomSolarize", 
# "RandomAdjustSharpness" 

def LoadTransformations(transform_config: str, device: str = "cpu"):
    tforms = []
    if "RandomVerticalFlip" in transform_config:
        tforms.append(RandVerticalFlip(0.5))
    if "RandomHorizontalFlip" in transform_config:
        tforms.append(RandHorizontalFlip(0.5))
    if "Rescale" in transform_config:
        rescale = transform_config["Rescale"]
        tforms.append(Rescale(rescale))
    if "RandomCrop" in transform_config:
        tforms.append(RandomCrop())
    if "RandomInvert" in transform_config:
        tforms.append(RandomInvert())
    if "RandomAutocontrast" in transform_config:
        tforms.append(RandomAutocontrast())
    if "RandomEqualize" in transform_config:
        tforms.append(RandomEqualize())
    if "Normalize" in transform_config:
        mode = transform_config["Normalize"]
        tforms.append(Normalize(mode))
    if "Standardize" in transform_config:
        tforms.append(Standardize())
    if "AdjustBrightness" in transform_config:
        rate = transform_config["AdjustBrightness"]["rate"]
        brightness = random.random()
        tforms.append(AdjustBrightness(rate, brightness))
    if "Blur" in transform_config:
        rate = transform_config["Blur"]["rate"]
        sigma = transform_config["Blur"]["sigma"]
        tforms.append(Blur(rate,sigma))
    if "ToTensor" in transform_config:
        tforms.append(ToTensor())
    transform = transforms.Compose(tforms)
    return transform


class RandVerticalFlip(object):

    def __init__(self, p):
        logger.info(
            f"Loaded RandomVerticalFlip transformation with probability {p}")
        self.p = p

    def __call__(self, sample):
        image = sample['image']
        flipped = False
        if random.random() < self.p:
            image = np.flip(image, axis=2)
            flipped = True
        sample["image"] = image
        sample["vertical_flip"] = flipped
        return sample


class RandHorizontalFlip(object):

    def __init__(self, p):
        logger.info(
            f"Loaded RandomHorizontalFlip transformation with probability {p}")
        self.p = p

    def __call__(self, sample):
        image = sample['image']
        flipped = False
        if random.random() < self.p:
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

    def __init__(self, mode="norm", min_scale = 0.0, max_scale = 255.0):
        if mode == "norm":
            logger.info(
                f"Loaded Normalize transformation that scales pixel values between 0 to 1")
        if mode == "sym":
            logger.info(
                f"Loaded Normalize transformation that scales pixel values between -1 to 1")
        self.mode = mode
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, sample):

        image = sample['image'].astype(np.float32)

        if self.mode == "norm":
            image -= image.min()
            image /= image.max()

        if self.mode == "sym":
            image = -1 + 2.0*(image - image.min())/(image.max() - image.min())
        sample["image"] = image
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device="cpu"):
        self.device = device
        logger.info(
            f"Loaded ToTensor transformation, putting tensors on device {device}")

    def __call__(self, sample):
        image = sample['image']
        size = len(image.shape)
        if size == 2:
            image = image.astype(np.float32)
            image = image.reshape(1, image.shape[0], image.shape[1])
        elif size == 3:
            image = image[:].astype(np.float32)
        sample["image"] = image #torch.from_numpy(image)
        return sample
    

class ColorJitter(torch.nn.Module):
    """Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "L", "I", "F" and modes with transparency (alpha channel) are not supported.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__()
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    @torch.jit.unused
    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness: Optional[List[float]],
                   contrast: Optional[List[float]],
                   saturation: Optional[List[float]],
                   hue: Optional[List[float]]
                   ) -> Tuple[Tensor, Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Get the parameters for the randomized transform to be applied on image.
        Args:
            brightness (tuple of float (min, max), optional): The range from which the brightness_factor is chosen
                uniformly. Pass None to turn off the transformation.
            contrast (tuple of float (min, max), optional): The range from which the contrast_factor is chosen
                uniformly. Pass None to turn off the transformation.
            saturation (tuple of float (min, max), optional): The range from which the saturation_factor is chosen
                uniformly. Pass None to turn off the transformation.
            hue (tuple of float (min, max), optional): The range from which the hue_factor is chosen uniformly.
                Pass None to turn off the transformation.
        Returns:
            tuple: The parameters used to apply the randomized transform
            along with their random order.
        """
        fn_idx = torch.randperm(4)

        b = None if brightness is None else float(torch.empty(1).uniform_(brightness[0], brightness[1]))
        c = None if contrast is None else float(torch.empty(1).uniform_(contrast[0], contrast[1]))
        s = None if saturation is None else float(torch.empty(1).uniform_(saturation[0], saturation[1]))
        h = None if hue is None else float(torch.empty(1).uniform_(hue[0], hue[1]))

        return fn_idx, b, c, s, h

    def __call__(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Input image.
        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        img = sample['image']
        
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

        sample["image"] = img
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

    
    
class RandomInvert(object):
    """Inverts the colors of the given image randomly with a given probability.
    If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".
    Args:
        p (float): probability of the image being color inverted. Default value is 0.5
    """

    def __init__(self, p=0.5): 
        super().__init__()
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be inverted.
        Returns:
            PIL Image or Tensor: Randomly color inverted image.
        """
        img = sample['image']
        
        if torch.rand(1).item() < self.p:
            img = np.invert(img)
            
        sample["image"] = img
        
        return sample
    
class RandomAutocontrast(object):
    """Inverts the colors of the given image randomly with a given probability.
    If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".
    Args:
        p (float): probability of the image being color inverted. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be autocontrasted.
        Returns:
            PIL Image or Tensor: Randomly autocontrasted image.
        """
        img = sample['image']
        
        if torch.rand(1).item() < self.p:
            img = ImageOps.autocontrast(img)
            
        sample["image"] = img
        
        return sample
    
class RandomEqualize(object):
    """Equalize the histogram of the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "P", "L" or "RGB".
    Args:
        p (float): probability of the image being equalized. Default value is 0.5
    """
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be equalized.
        Returns:
            PIL Image or Tensor: Randomly equalized image.
        """
        img = sample['image']
        
        if torch.rand(1).item() < self.p:
            img = ImageOps.equalize(img)
            
        sample["image"] = img
        
        return sample
    
class AdjustBrightness(object):
    
    def __init__(self, rate=0.5, brightness_factor=1):
        super().__init__()
        self.rate = rate
        self.brightness_factor = brightness_factor
    
    def __call__(self, sample):
        """Adjust brightness of an Image.

        Args:
            img (np.ndarray): CV Image to be adjusted.
            brightness_factor (float):  How much to adjust the brightness. Can be
                any non negative number. 0 gives a black image, 1 gives the
                original image while 2 increases the brightness by a factor of 2.

        Returns:
            np.ndarray: Brightness adjusted image.
        """
        if random.uniform(0, 1) <= self.rate:
            image = sample['image']
            im = image.astype(np.float32) * self.brightness_factor
            im = im.clip(min=0, max=1)
            sample['image'] = im.astype(image.dtype)    
        return sample
        
class Blur(object):
    
    def __init__(self, rate=0.5, sigma=1):
        super().__init__()
        self.rate = rate
        self.sigma = sigma
    
    def __call__(self, sample):
        if random.uniform(0, 1) <= self.rate:
            image = sample['image']
            image = gaussian_filter(
                image, 
                sigma = self.sigma
            )
            sample['image'] = image
        return sample
