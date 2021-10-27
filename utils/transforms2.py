from torchvision import transforms
from datetime import datetime
import torch
import cv2
import os
import sys
import math
from pathlib import Path
from tqdm.auto import tqdm
import random
import numpy as np
from matplotlib import pyplot as plt
import utils.utilities as utilities
from functools import partial
import torchvision.transforms.functional as F

# Imports for mask geneation
from scipy.spatial import ConvexHull
from heart_echo.Helpers import LABELTYPE
from utils.constants import TRAIN_PATIENT_IDS, VAL_PATIENT_IDS, TEST_PATIENT_IDS
import multiprocessing as mp
from heart_echo.pytorch import HeartEchoDataset

FILL_VAL = 0.3


class Identity():
    """
    Identity transformation
    """

    def __call__(self, sample):
        return sample


class VideoSubsample():
    """
    Randomly sample specified number of frames from video
    """

    def __init__(self, num_frames):
        self.num_frames = num_frames

    def __call__(self, sample):
        sample, p_id = sample
        T = len(sample)
        start = random.randint(0, T - 2 * self.num_frames + 1)
        frames = []
        for i in range(self.num_frames):
            frames.append(sample[start + 2 * i].squeeze())
        return frames, p_id


class ConvertToTensor():
    """
    Convert numpy array to Tensor
    """

    def __call__(self, sample):
        sample, p_id = sample
        if isinstance(sample, list):
            return (torch.stack([torch.from_numpy(s).float().unsqueeze(0) for s in sample]), p_id)
        else:
            return (torch.from_numpy(sample).float().unsqueeze(0), p_id)


class ShapeEqualization():
    """
    Warp each echo into uniform shape
    """

    def _get_masks(self):
        mask_path = os.path.expanduser(os.path.join('~', '.echo-net', 'masks'))
        mask_fn = os.path.join(mask_path, f'{-1}_{int(100 * self.orig_img_scale)}_percent.pt')
        if not os.path.exists(mask_fn):
            utilities.generate_masks(-1, self.orig_img_scale)
        return torch.load(mask_fn)

    def _generate_arc_points(self, left, middle, right, n_parts=1000):
        """
        Take three points defining an arc and generate n_parts equispaced points on it
        """
        # Get circle defining the arc
        radius, center = utilities.circle_from_points(left, middle, right)

        # Shift points such that center is at origin
        centered_left = left - center
        centered_right = right - center

        # Calculate angles of left and right point
        angle_left = math.atan2(centered_left[1], centered_left[0])
        angle_right = math.atan2(centered_right[1], centered_right[0])

        # Generate angle for distance between arc points
        part = (angle_right - angle_left) / n_parts

        # Generate points on arc
        points = []
        for i in range(n_parts):
            theta = i * part + angle_left
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            point = np.array([x, y]) + center
            points.append(point)
        return np.array(points)

    def _generate_line_points(self, p1, p2, n_parts=400):
        """
        Generate n_parts equispaced points on line segment defined over p1 and p2
        """
        points = []
        for i in range(n_parts):
            point = p1 + (p2 - p1) * float(i) / n_parts
            points.append(point)
        return np.array(points)

    def _generate_perspective_transforms(self, size):
        # Get echo mask for each patient
        mask_path = os.path.expanduser(os.path.join('~', '.echo-net', 'masks'))
        mask_fn = os.path.join(mask_path, f'{-1}_{int(100 * self.orig_img_scale)}_percent.pt')
        masks = self._get_masks()

        fn = os.path.join(self.transform_path, f'{size}.pt')
        # Get transformation for each patient
        print("Assembling perspective transformation for shape equalization.")
        transforms = {}
        for patient in tqdm(TRAIN_PATIENT_IDS + VAL_PATIENT_IDS + TEST_PATIENT_IDS):
            # p_id = f'{patient}CV'
            p_id = f'{patient}KAPAP'
            # Get high resolution mask
            if not p_id in masks:
                continue
            mask = masks[p_id]

            # Assemble initial point correspondences
            pa = utilities.get_arc_points_from_mask(mask)
            pb = np.array([[size // 32, size / 2], [6 * size / 8, size - size // 32], [size - size // 32, size / 2],
                           [6 * size / 8, size // 32]], dtype=np.int32)

            # Add point correspondences on arc
            left, middle, right = pa[3], pa[2], pa[1]
            pa = np.concatenate((pa, self._generate_arc_points(left, middle, right)))
            left, middle, right = pb[3], pb[2], pb[1]
            pb = np.concatenate((pb, self._generate_arc_points(left, middle, right)))

            # Add point correspondences on line segments
            p1, p2 = pa[0], pa[1]
            pa = np.concatenate((pa, self._generate_line_points(p1, p2)))
            p1, p2 = pa[0], pa[3]
            pa = np.concatenate((pa, self._generate_line_points(p1, p2)))

            p1, p2 = pb[0], pb[1]
            pb = np.concatenate((pb, self._generate_line_points(p1, p2)))
            p1, p2 = pb[0], pb[3]
            pb = np.concatenate((pb, self._generate_line_points(p1, p2)))

            # Compute homography and add to transformation dict
            pa, pb = np.float32(pa), np.float32(pb)
            homog, _ = cv2.findHomography(np.array([pa[:, 1], pa[:, 0]]).T, np.array([pb[:, 1], pb[:, 0]]).T,
                                          cv2.RANSAC)
            transforms[p_id] = homog

        torch.save(transforms, fn)

    def _get_perspective_transforms(self, size):
        fn = os.path.join(self.transform_path, f'{size}.pt')
        if not os.path.exists(fn):
            self._generate_perspective_transforms(size)
        return torch.load(fn)

    def __init__(self, resize=256, orig_img_scale=0.5):
        self.resize = resize
        self.orig_img_scale = orig_img_scale
        self.transform_path = os.path.expanduser(os.path.join('~', '.echo-net', 'shape_equalization'))
        Path(self.transform_path).mkdir(parents=True, exist_ok=True)
        self.perspective_transforms = self._get_perspective_transforms(resize)

    def __call__(self, sample):
        sample, p_id = sample
        sample = sample.squeeze()
        homog = self.perspective_transforms[p_id]
        sample = cv2.warpPerspective(np.float32(sample), homog, (self.resize, self.resize))
        sample = torch.from_numpy(sample).float().unsqueeze(0)
        return sample


class RandomMask():
    """
    Apply a random arc mask from any patient
    """

    def _get_masks(self, resize):
        mask_fn = os.path.join(self.mask_path, f'{resize}_{int(100 * self.orig_img_scale)}_percent.pt')
        if not os.path.exists(mask_fn):
            utilities.generate_masks(resize, self.orig_img_scale)
        return torch.load(mask_fn)

    def __init__(self, resize=256, orig_img_scale=0.5):
        self.orig_img_scale = orig_img_scale
        self.mask_path = os.path.expanduser(os.path.join('~', '.echo-net', 'masks'))
        Path(self.mask_path).mkdir(parents=True, exist_ok=True)
        self.masks = self._get_masks(resize)

    def __call__(self, sample):
        mask_keys = list(self.masks.keys())
        random_mask_idx = random.randint(0, len(mask_keys) - 1)
        random_mask = self.masks[mask_keys[random_mask_idx]]
        return sample * random_mask


class MinMask():
    """
    Apply the minimum pixel arc mask to sample
    """

    def _get_masks(self, resize):
        mask_fn = os.path.join(self.mask_path, f'{resize}_{int(100 * self.orig_img_scale)}_percent.pt')
        if not os.path.exists(mask_fn):
            utilities.generate_masks(resize, self.orig_img_scale)
        return torch.load(mask_fn)

    def _find_min_mask(self):
        n_nonzero = torch.Tensor([torch.nonzero(mask).shape[0] for mask in self.masks.values()])
        min_idx = n_nonzero.argmin()
        self.min_mask = list(self.masks.values())[min_idx]

    def __init__(self, resize=256, orig_img_scale=0.5):
        self.orig_img_scale = orig_img_scale
        self.mask_path = os.path.expanduser(os.path.join('~', '.echo-net', 'masks'))
        Path(self.mask_path).mkdir(parents=True, exist_ok=True)
        self.masks = self._get_masks(resize)
        self._find_min_mask()

    def __call__(self, sample):
        return sample * self.min_mask


class CropToCorners():
    """
    Crop echo to have the its four corners at the border
    """

    def _get_masks(self, index_file_path):
        mask_fn = os.path.join(self.mask_path, f'{-1}_{int(100 * self.orig_img_scale)}_percent.pt')
        if not os.path.exists(mask_fn):
            gen_masks(-1, self.orig_img_scale, index_file_path)
            # utilities.generate_masks(-1, self.orig_img_scale)
        return torch.load(mask_fn)

    def _get_mask_corners(self):
        corners_fn = os.path.join(self.corner_path, f'{int(100 * self.orig_img_scale)}_percent.pt')
        if not os.path.exists(corners_fn):
            utilities.generate_mask_corners(self.orig_img_scale)
        return torch.load(corners_fn)

    def __init__(self, index_file_path, orig_img_scale=0.5):
        self.orig_img_scale = orig_img_scale

        # Generate precomputation libraries for masks and corners
        self.mask_path = os.path.expanduser(os.path.join('~', '.echo-net', 'masks'))
        self.corner_path = os.path.expanduser(os.path.join('~', '.echo-net', 'mask_corners'))
        Path(self.mask_path).mkdir(parents=True, exist_ok=True)
        Path(self.corner_path).mkdir(parents=True, exist_ok=True)

        # Get masks and corners
        self.masks = self._get_masks(index_file_path)
        self.corners = self._get_mask_corners()

    def __call__(self, sample):
        sample, p_id = sample
        T, R, B, L = self.corners[p_id]
        sample = sample[:, T[0]:B[0], L[1]:R[1]]
        return (sample, p_id)


class HistEq():
    """
    Perform Histogram Equalization
    """

    def __call__(self, sample):
        sample, p_id = sample
        if isinstance(sample, list):
            sample = [cv2.equalizeHist(s) for s in sample]
        else:
            sample = cv2.equalizeHist(sample)
        return (sample, p_id)


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
        sample = transforms.functional.resize(sample, size=(int(rand_scale * H), int(rand_scale * W)))
        sample = transforms.functional.center_crop(sample, output_size=(H, W))
        return sample


class RandResizePad():
    """
    Randomly resize image to given scale and pad to arrive back at original scale
    """

    def __init__(self, scale, pad_noise=False):
        assert scale < 1, "Scale must be greater than 1 for RandResizeCrop"
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
        sample = transforms.functional.pad(sample, padding=(pad_left_right, pad_up_down), fill=FILL_VAL)
        assert H - 2 <= sample.shape[-2] <= H + 2 and W - 2 <= sample.shape[
            -1] <= W + 2, f"Wrong dimension after padding, original was {(H, W)}, new is {sample.shape[-2:]}"
        if self.pad_noise:
            samples = self._pad_noise(sample, pad_up_down, pad_left_right)
        sample = transforms.functional.resize(sample, size=(H, W))
        return sample


class Augment():
    """
    Randomly perform a range of data augmentation
    including translation, rotation, scaling, salt & pepper noise,
    brightness adjustment, Gamma Correction, blurring and sharpening the image
    """

    def __init__(self, index_file_path, orig_img_scale=0.5, size=-1, return_pid=False, fold=0, valid=False):
        self.return_pid = return_pid
        # Generate precomputation libraries for masks and corners
        if valid:
            subset = 'valid'
        else:
            subset = 'train'
        self.mask_path = os.path.expanduser(os.path.join('~', '.echo-net', 'masks', subset))
        Path(self.mask_path).mkdir(parents=True, exist_ok=True)

        # Get masks and corners
        self.orig_img_scale = orig_img_scale
        self.index_file_path = index_file_path
        self.size = size
        self.masks = self._get_masks(fold)

        # Define augmentation transforms
        self.intensity_transformations = [
            RandomSharpness(),
            RandomBrightnessAdjustment(),
            RandomGammaCorrection(),
            SaltPepperNoise()
        ]
        self.positional_transformations = [
            None,
            transforms.RandomAffine(0, translate=(0.1, 0.1), fill=FILL_VAL),
            RandomResize(),
        ]

    def _get_masks(self, fold):
        print('in _get_masks')
        mask_fn = os.path.join(self.mask_path, f'{self.size}_{int(100 * self.orig_img_scale)}_percent_fold{fold}.pt')
        print(mask_fn)
        if not os.path.exists(mask_fn):
            # utilities.generate_masks(self.size, self.orig_img_scale)
            gen_masks(self.size, self.orig_img_scale, self.index_file_path)
        return torch.load(mask_fn)

    def _apply_background_noise(self, sample, mask):
        background = FILL_VAL * torch.ones(sample.shape)
        background = background.to(sample.device)
        if len(sample.shape) == 4 and len(mask.shape) < 4:
            sample[:, mask == 0] = sample[:, mask == 0] + background[:, mask == 0]
        else:
            sample[mask == 0] = sample[mask == 0] + background[mask == 0]
        return sample

    def _apply_mask(self, sample, mask):
        return mask * sample

    def _cut_border(self, sample, mask):
        H, W = sample.shape[-2], sample.shape[-1]
        sample = transforms.functional.resize(sample, size=(int(1.05 * H), int(1.05 * W)))
        sample = transforms.functional.center_crop(sample, output_size=(H, W))
        return sample * mask

    def _add_background_speckle_noise(self, sample):
        std = 0.3
        sample[sample == FILL_VAL] = sample[sample == FILL_VAL] + \
                                     std * sample[sample == FILL_VAL] * torch.randn_like(sample[sample == FILL_VAL])
        sample = torch.clamp(sample, min=0, max=1)
        return sample

    def __call__(self, sample):
        # Get sample and corresponding mask
        sample, p_id = sample
        if torch.rand(1) < 0.25:  # 25 % of images don't get any augmentation
            if self.return_pid:
                return (sample, p_id)
            return sample
        # If processing multiple samples, assemble mask tensor and reshape samples
        if isinstance(p_id, tuple):
            B, T, _, H, W = sample.shape
            sample = sample.reshape(-1, 1, H, W)
            mask = torch.cat([self.masks[id].unsqueeze(0) for id in p_id])
            mask = mask.reshape(B, 1, H, W).repeat(1, T, 1, 1).reshape(-1, 1, H, W)
        else:
            mask = self.masks[p_id].unsqueeze(0)

        # Try moving mask to gpu if available
        mask = mask.to(sample.device)

        # Apply intensity transformations
        for t in self.intensity_transformations:
            # if 0.7 < torch.rand(1):
            if torch.rand(1) < 0.5:
                sample = t(sample)

        # Cut off black border around echo
        sample = self._cut_border(sample, mask)
        # Add background noise
        sample = self._apply_background_noise(sample, mask)

        # Define Random Rotation around top corner
        if self.positional_transformations[0] is None:
            rot_center = (sample.shape[1] // 2, 0)
            rand_rot = transforms.RandomRotation(15, fill=FILL_VAL, center=rot_center)
            self.positional_transformations[0] = rand_rot

        # Apply positional transformations
        for t in self.positional_transformations:
            # if 0.7 < torch.rand(1):
            if torch.rand(1) < 0.5:
                sample = t(sample)

        # Retrieve original shape
        # sample = self._apply_mask(sample, mask)

        # Add speckle noise to background
        sample = self._add_background_speckle_noise(sample)

        # Reshape sample back to batched timeseries in case of batch augmentation
        if isinstance(p_id, tuple):
            sample = sample.reshape(B, T, 1, H, W)

        if self.return_pid:
            return (sample, p_id)
        return sample


class RandomResize():
    """
    Randomly scale and crop or scale and pad the image
    """

    def __init__(self, pad_noise=False):
        self.transforms = [
            # RandResizeCrop(1.4),
            RandResizeCrop(1.2),
            # RandResizePad(0.6, pad_noise),
            RandResizePad(0.8, pad_noise)
        ]

    def __call__(self, sample):
        rand_transform = random.randint(0, 1)
        return self.transforms[rand_transform](sample)


class Normalize():
    """
    Standardize input image
    """

    def __call__(self, sample):
        sample, p_id = sample
        return (sample.float() / 255., p_id)


class SaltPepperNoise():
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
        sample = F.adjust_gamma(sample, gamma=rand_gamma)
        return sample


class RandomSharpness():
    """
    Randomly increase oder decrease image sharpness
    """

    def __call__(self, sample):
        if 0.5 < torch.rand(1):
            rand_factor = random.random() * 7 + 1
        else:
            rand_factor = random.random()
        sample = F.adjust_sharpness(sample, sharpness_factor=rand_factor)
        return sample


class RandomNoise():
    """
    Add Gaussian noise to the image
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


class Resize():
    """
    Resize image to given size, -1 for original size
    """

    def __init__(self, size, return_pid=True):
        self.return_pid = return_pid
        if size == -1:
            self.transform = Identity()
        else:
            self.transform = transforms.Resize((size, size))

    def __call__(self, sample):
        sample, p_id = sample
        if self.return_pid:
            return self.transform(sample), p_id
        else:
            return self.transform(sample)


def get_transforms(
        index_file_path,
        fold=0,
        valid=False,
        resize=256,
        crop_to_corner=False,
        shape_equalization=False,
        noise=False,
        mask=False,
        min_mask=False,
        augment=True,
        with_pid=False,
        dataset_orig_img_scale=0.25,
        num_video_frames=None
):
    """
    Compose a set of prespecified transformation using the torchvision transform compose class
    """
    assert not (
                crop_to_corner and shape_equalization), "Cannot do crop_to_corner and shape_equalization simultaneously."
    if shape_equalization and (mask or min_mask):
        print("Ignoring masking transformations and random resizing because shape_equalization equals True.")
    return transforms.Compose(
        [
            VideoSubsample(num_video_frames) if num_video_frames is not None else Identity(),
            HistEq(),
            ConvertToTensor(),
            Normalize(),
            CropToCorners(orig_img_scale=dataset_orig_img_scale) if crop_to_corner else Identity(),
            ShapeEqualization(resize, orig_img_scale=dataset_orig_img_scale) if shape_equalization else Identity(),
            #             GaussianSmoothing(),
            Resize(resize, return_pid=(with_pid or augment)) if not shape_equalization else Identity(),
            Augment(index_file_path, orig_img_scale=dataset_orig_img_scale, size=resize, return_pid=with_pid, fold=fold,
                    valid=valid)
            if augment else Identity(),
            RandomNoise() if noise else Identity(),
            RandomMask(resize=resize,
                       orig_img_scale=dataset_orig_img_scale) if mask and not shape_equalization else Identity(),
            MinMask(resize=resize,
                    orig_img_scale=dataset_orig_img_scale) if min_mask and not shape_equalization else Identity(),
        ]
    )


def gen_masks(resize, orig_scale_fac, index_file_path, fold=0):
    print('in get_masks')
    mask_path = os.path.expanduser(os.path.join('~', '.echo-net', 'masks'))
    mask_fn = os.path.join(mask_path,
                           f'{resize}_{int(100 * orig_scale_fac)}_percent_fold{fold}.pt')
    # Get a mask for each patient
    print("Assembling echo masks.")
    masks = {}
    p_ids = [str(id) + 'KAPAP' for id in np.load(index_file_path)]
    results = []
    # for p_id in p_ids:
    # result = _gen_mask(index_file_path, p_id, resize, orig_scale_fac)
    with mp.Pool(processes=16) as pool:
        # for result in pool.map(_gen_mask, p_ids):
        for result in pool.map(partial(_gen_mask, index_file_path, resize, orig_scale_fac), p_ids):
            results.append(result)
    # Join results
    for tup in results:
        if tup is not None:
            p_id = tup[0]
            masks[p_id] = tup[1]
    torch.save(masks, mask_fn)


def _gen_mask(index_file_path, resize, orig_scale_fac, p_id):
    """
    Generate mask for p_id
    """
    transform = get_transforms(index_file_path, resize=resize, augment=False)
    cache_dir = os.path.expanduser('~/.heart_echo')  # TODO: Make parameter !
    curr_video_path = os.path.join(cache_dir, str(orig_scale_fac), str(p_id) + '.npy')
    if not os.path.exists(curr_video_path):  # Skip unavailable patients
        return
    print('CREATING MASK FOR', curr_video_path)
    frames = np.load(curr_video_path)  # load numpy video frames
    frames = [transform((frame, p_id)).unsqueeze(0) for frame in frames]
    # Extract mask from patient
    H, W = frames[0][0].shape[1:]
    # H, W = frames[0].shape
    vid = torch.cat(tuple(frame[0] for frame in frames), dim=0)
    mask = torch.std(vid, dim=0)
    mask[mask < 0.01] = 0

    # Remove pixels without surrounding pixels
    nonzero = torch.nonzero(mask)
    for idx in nonzero:
        if idx[0] + 1 < H and idx[1] + 1 < W and \
                idx[0] - 1 >= 0 and idx[1] - 1 >= W and \
                mask[idx[0] + 1, idx[1]] == 0 and \
                mask[idx[0] - 1, idx[1]] == 0 and \
                mask[idx[0], idx[1] + 1] == 0 and \
                mask[idx[0], idx[1] - 1] == 0:
            mask[idx[0], idx[1]] = 0


    # Augment mask with convex hull
    hull = ConvexHull(torch.nonzero(mask))
    hull_mask = torch.ones(mask.shape)
    for i in range(hull_mask.shape[0]):
        for j in range(hull_mask.shape[1]):
            point = np.array([i, j, 1])
            for eq in hull.equations:
                if eq.dot(point) > 0:
                    hull_mask[i, j] = 0

    return p_id, hull_mask
