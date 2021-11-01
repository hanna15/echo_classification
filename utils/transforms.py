from torchvision import transforms
import torch
import cv2
import math
import random
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode as i_mode
import numpy as np

FILL_VAL = 0.3  # ?? Where does this value come from (??)


class HistEq():
    """
    Perform Histogram Equalization
    """

    def __call__(self, sample):
        sample = cv2.equalizeHist(sample)
        return sample


class RandResizeCrop():
    """
    Randomly resize image to given scale and crop it back to original scale
    """

    def __init__(self, scale):
        assert scale > 1, "Scale must be greater than 1 for RandResizeCrop"
        self.max_scale = scale

    def __call__(self, sample):
        H, W = sample.shape[-2], sample.shape[-1]
        rand_scale = random.random() * (self.max_scale - 1) + 1
        sample = transforms.functional.resize(sample, size=[int(rand_scale * H), int(rand_scale * W)])
        sample = transforms.functional.center_crop(sample, output_size=[H, W])
        return sample


class RandResizePad():
    """
    Randomly resize image to given scale and pad to arrive back at original scale
    """

    def __init__(self, scale, pad_noise=False):
        assert scale < 1, "Scale must be less than 1 for RandResizePad"
        self.pad_noise = pad_noise
        self.min_scale = scale

    def _pad_noise(self, sample, pad_up_down, pad_left_right):
        background = torch.rand(sample.shape) + sample
        background[sample != 0] = 0
        sample[:, :pad_up_down, :] = background[:, :pad_up_down, :]
        sample[:, -pad_up_down:, :] = background[:, -pad_up_down:, :]
        sample[:, :, :pad_left_right] = background[:, :, :pad_left_right]
        sample[:, :, -pad_left_right:] = background[:, :, -pad_left_right:]
        return sample

    def __call__(self, sample):
        H, W = sample.shape[-2], sample.shape[-1]
        rand_scale = random.random() * (1 - self.min_scale) + self.min_scale
        sample = transforms.functional.resize(sample, size=(int(rand_scale * H), int(rand_scale * W)))
        new_H, new_W = sample.shape[-2], sample.shape[-1]
        pad_up_down, pad_left_right = (H - new_H) // 2, (W - new_W) // 2
        # sample = transforms.functional.pad(sample, padding=(pad_left_right, pad_up_down), fill=FILL_VAL) # This will lead to all-black img
        sample = transforms.functional.pad(sample, padding=[pad_left_right, pad_up_down]) # Using default fill value works
        assert H - 2 <= sample.shape[-2] <= H + 2 and W - 2 <= sample.shape[
            -1] <= W + 2, f"Wrong dimension after padding, original was {(H, W)}, new is {sample.shape[-2:]}"
        if self.pad_noise:
            samples = self._pad_noise(sample, pad_up_down, pad_left_right)
        sample = transforms.functional.resize(sample, size=(H, W))
        return sample


class Rotate():
    def __call__(self, sample):
        deg = random.uniform(-5, 5)  # Up to 5 degrees
        sample = transforms.functional.rotate(sample, angle=deg)
        return sample


class Translate():
    def __call__(self, sample):
        transl = random.randint(-5, 5)  # Up to 5 degrees
        sample = transforms.functional.affine(sample, angle=0, translate=[transl, transl], scale=1.0, shear=0.0)
        return sample


class Intensity():
    """
    Randomly perform a range of data augmentation
    including translation, rotation, scaling, salt & pepper noise,
    brightness adjustment, Gamma Correction, blurring and sharpening the image
    """

    def __init__(self):

        # Define augmentation transforms
        self.intensity_transformations = [
            RandomSharpness(),
            RandomBrightnessAdjustment(),
            RandomGammaCorrection(),
            SaltPepperNoise()
        ]

    def __call__(self, sample):
        # Get sample and corresponding mask
        # If processing multiple samples, assemble mask tensor and reshape samples

        # Apply intensity transformations
        for t in self.intensity_transformations:
            # Apply each transformation with 50 % possibility
            # So, could have all or no transformations, but usually only 1-2 transformations will be picked
            if torch.rand(1) > 0.5:
                sample = t(sample)
        return sample


class RandomResize():
    """
    Randomly scale and crop or scale and pad the image
    """

    def __init__(self, pad_noise=False):
        self.transforms = [
            RandResizeCrop(1.15),  # max scale up 1.15x
            RandResizePad(0.85, pad_noise),  # max scale down by 0.85x
        ]

    def __call__(self, sample):
        rand_transform = random.randint(0, 1)
        return self.transforms[rand_transform](sample)


class Normalize():
    """
    Standardize input image
    """

    def __call__(self, sample):
        return sample.float() / 255.


class SaltPepperNoise(): # TODO: Look into this - does not work as planned (!)
    """
    Add Salt and Pepper noise on top of the image
    """

    def __init__(self, thresh=0.005):
        self.thresh = thresh

    def __call__(self, sample):
        noise = torch.rand(sample.shape)
        sample[noise < self.thresh] = 0
        sample[noise > 1 - self.thresh] = 1
        return sample


class RandomBrightnessAdjustment():
    """
    Randomly adjust brightness
    """

    def __call__(self, sample):
        rand_factor = random.random() * 0.7 + 0.5
        sample = F.adjust_brightness(sample, brightness_factor=rand_factor)
        return sample


class RandomGammaCorrection():
    """
    Do Gamma Correction with random gamma as described in
    https://en.wikipedia.org/wiki/Gamma_correction
    """

    def __call__(self, sample):
        rand_gamma = random.random() * 1.75 + 0.25
        # Below 0 => makes shadows brighter. Above 0 => makes shadows darker.
        # rand_gamma = random.uniform(0.5, 1.5)  # I prefer not larger range, to keep 'normal' looking
        sample = F.adjust_gamma(sample, gamma=rand_gamma)
        return sample


class RandomSharpness():
    """
    Randomly increase or decrease image sharpness
    """

    def __call__(self, sample):
        if 0.5 < torch.rand(1):  # 50 % increase sharpness (e.g. 2 increases sharpness by 2)
            rand_factor = random.random() * 7 + 1
            # rand_factor = random.uniform(1, 4)  # max 4x sharpness
        else:  # 50% of the time, reduce sharpness (by providing number < 1)
            # 0 gives a blurred image, 1 gives original image
            rand_factor = random.random()  # num between [0, 1( ==> but in reality no change of exact 0
        sample = F.adjust_sharpness(sample, sharpness_factor=rand_factor)
        return sample


class RandomNoise():
    """
    Add Gaussian noise to the image. Must be called before normalization
    """

    def __init__(self):
        # Standarddeviation of noise is 5 pixel intensities
        self.std = 5. / 255.

    def __call__(self, sample):
        return torch.clamp(sample + self.std * torch.randn_like(sample), min=0, max=1)


class GaussianSmoothing():
    """
    Do Gaussian filtering with low sigma to smooth out edges
    """

    def __init__(self, kernel_size=11, sigma=0.5):
        # Define Gaussian Kernel
        self.gaussian_kernel = self._generate_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)

    def _generate_gaussian_kernel(self, kernel_size, sigma):
        """
        Code from https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/3
        """

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1)

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                              (2 * variance)
                          )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Set pytorch convolution from gaussian kernel
        gaussian = torch.nn.Conv2d(1, 1, kernel_size, padding=kernel_size // 2, padding_mode='zeros', bias=False)
        gaussian.weight.requires_grad = False
        gaussian.weight[:, :] = gaussian_kernel
        return gaussian

    def __call__(self, sample):
        sample, p_id = sample
        # Slightly smooth out edges with Gaussian Kernel
        with torch.no_grad():
            if len(sample.shape) == 3:
                sample = self.gaussian_kernel(sample.unsqueeze(0)).squeeze(0)
            else:
                sample = self.gaussian_kernel(sample)
        return sample, p_id


# class LaplaceNoise():
#     def __init(self, mean=0.0, std=1.0):
#         self.mean = mean
#         self.std = std
#
#     def __call__(self, sample):
#         return sample + np.random.laplace() * self.std + self.mean


def get_base_transforms(hist_eq=True):
    augment_list = []
    if hist_eq:
        augment_list.append(HistEq())
    augment_list.extend(get_list_of_base_transforms_except_norm())
    augment_list.append(Normalize())
    return transforms.Compose(augment_list)


def get_list_of_base_transforms_except_norm():
    return [transforms.ToPILImage(),
            # transforms.Resize(size=(128, 128), interpolation=i_mode.BICUBIC),
            transforms.Resize(size=(224, 224), interpolation=i_mode.BICUBIC),
            transforms.ToTensor()]


def get_augment_transforms(hist_eq=True, noise=True, intensity=True, rand_resize=True, rotate=True, translate=True):
    augment_list = []
    if hist_eq:
        augment_list.append(HistEq())
    augment_list.extend(get_list_of_base_transforms_except_norm())  # base transforms that must always be present
    if intensity:
        augment_list.append(transforms.RandomApply([Intensity()], 0.65))
    if noise:
        augment_list.append(transforms.RandomApply([RandomNoise()]))
    if rand_resize:
        augment_list.append(transforms.RandomApply([RandomResize()]))
    if rotate:
        augment_list.append(transforms.RandomApply([Rotate()]))
    if translate:
        augment_list.append(transforms.RandomApply([Translate()]))
    augment_list.append(Normalize())  # Always normalize
    return transforms.Compose(augment_list)
