from torchvision import transforms
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
import multiprocessing as mp
from itertools import repeat
import time

# Geometry
from sympy import Line, Circle

# Imports for mask generation
from scipy.spatial import ConvexHull
from sklearn.linear_model import RANSACRegressor, LogisticRegression
from heart_echo.Helpers import LABELTYPE
from utils.constants import *
from heart_echo.pytorch import HeartEchoDataset
# from utils.transforms import get_transforms


def get_datasets(resize=256,
                 crop_to_corner=False,
                 shape_equalization=False,
                 noise=False,
                 mask=False,
                 min_mask=False,
                 augment=True,
                 with_pid=False,
                 frame_block_size=1,
                 num_video_frames=None,
                 data_split=0,
                 label_type='structural',
                 view=None
                 ):
    return # TODO: Change back to normal - temporally commented out bc of circular reference
    # # Define image scale on which to precompute echos
    # img_scale = 0.25
    #
    # # Define data transforms that should be applied to dataset
    # train_transform = get_transforms(
    #     resize=resize,
    #     crop_to_corner=crop_to_corner,
    #     shape_equalization=shape_equalization,
    #     noise=noise,
    #     mask=mask,
    #     min_mask=min_mask,
    #     dataset_orig_img_scale=img_scale,
    #     augment=augment,
    #     with_pid=with_pid,
    #     num_video_frames=num_video_frames
    # )
    #
    # test_transform = get_transforms(
    #     resize=resize,
    #     crop_to_corner=crop_to_corner,
    #     shape_equalization=shape_equalization,
    #     noise=False,
    #     mask=False,
    #     augment=False,
    #     min_mask=min_mask,
    #     dataset_orig_img_scale=img_scale,
    #     with_pid=with_pid,
    #     num_video_frames=num_video_frames
    # )
    #
    # if label_type == 'structural':
    #     # Make sure split is between 0 and 4
    #     assert data_split in range(5), "data_split should be between 0 and 4"
    #     assert view is None or view == 'CV', "Currently only CV view is supported for structural labels"
    #
    #     # Get 4 CV
    #     train_ids = TRAIN_PATIENT_IDS
    #     val_ids = VAL_PATIENT_IDS
    #     test_ids = TEST_PATIENT_IDS
    #
    #     if data_split > 0:
    #         seeds = [0, 124123, 584731, 381734, 483823, 218232]
    #         healthy_ids = TRAIN_PATIENT_IDS + VAL_PATIENT_IDS + HEALTHY_TEST_PATIENT_IDS_CV
    #         train_test_split = torch.utils.data.random_split(
    #             healthy_ids,
    #             [len(TRAIN_PATIENT_IDS), len(VAL_PATIENT_IDS), len(HEALTHY_TEST_PATIENT_IDS_CV)],
    #             generator=torch.Generator().manual_seed(seeds[data_split])
    #         )
    #         train_ids = [id for id in train_test_split[0]]
    #         val_ids = [id for id in train_test_split[1]]
    #         test_ids = [id for id in train_test_split[2]] + ANO_TEST_PATIENT_IDS_CV
    # elif label_type == 'ph':
    #     if view is None:
    #         view = 'KAPAP'
    #     elif view not in ['KAPAP', 'KAAP', 'LA']:
    #         raise ValueError(f'View {view} is invalid for labeltype ph')
    #     seed = 419371
    #     train_test_split = torch.utils.data.random_split(
    #         PH_HEALTHY[view],
    #         [len(PH_HEALTHY[view]) - 14, 4, 10],
    #         generator=torch.Generator().manual_seed(seed)
    #     )
    #     train_ids = [id for id in train_test_split[0]]
    #     val_ids = [id for id in train_test_split[1]]
    #     test_ids = [id for id in train_test_split[2]]
    # else:
    #     raise ValueError('label_type should either be "structural" or "ph"')
    #
    # # Get datasets
    # echo_train = HeartEchoDataset(train_ids,
    #                               label_type=LABELTYPE.VISIBLE_FAILURE_STRONG,
    #                               transform=train_transform,
    #                               scale_factor=img_scale,
    #                               frame_block_size=-1 if num_video_frames is not None else frame_block_size,
    #                               procs=32)
    # echo_val = HeartEchoDataset(val_ids,
    #                             label_type=LABELTYPE.VISIBLE_FAILURE_STRONG,
    #                             transform=test_transform,
    #                             scale_factor=img_scale,
    #                             frame_block_size=-1 if num_video_frames is not None else frame_block_size,
    #                             procs=32)
    # echo_test = HeartEchoDataset(test_ids,
    #                              label_type=LABELTYPE.VISIBLE_FAILURE_STRONG,
    #                              transform=test_transform,
    #                              scale_factor=img_scale,
    #                              frame_block_size=-1 if num_video_frames is not None else frame_block_size,
    #                              procs=32)
    #
    # return echo_train, echo_val, echo_test


def get_img_datasets(**kwargs):
    return get_datasets(frame_block_size=1, **kwargs)


def get_video_datasets(frames=5, **kwargs):
    datasets = get_datasets(frame_block_size=frames, **kwargs)
    if frames == 1:
        for dataset in datasets:
            dataset.transform.transforms.append(lambda x: x.unsqueeze(0))
    return datasets


def reparametrization_trick(means, log_sigmas):
    dist = torch.distributions.normal.Normal(means, torch.exp(log_sigmas / 2))
    return dist.rsample()


def compute_kl_div(mus, log_vars, other_mus=torch.tensor(0), other_log_vars=torch.tensor(0)):
    """
    Compute KL Divergence between two Gaussians
    """
    kl_div = 0.5 * torch.sum(
        1. / torch.exp(other_log_vars) * torch.exp(log_vars) + (other_mus - mus) * 1. / torch.exp(other_log_vars) * (
                    other_mus - mus) - 1. + other_log_vars - log_vars, dim=1)
    assert torch.all(kl_div >= 0), print('KL div attains value smaller than zero!')
    return kl_div


def add_speckle_noise(sample):
    with torch.no_grad():
        std = 0.1
        sample = torch.clamp(sample + std * sample * torch.randn_like(sample), min=0, max=1)
    return sample


def _compute_increasing_weight(d):
    B, C, H, W = d.shape
    horizontal = torch.arange(W).repeat(B, C, H, 1).type_as(d) - W // 2

    # If width is 128, weight is 4 at 20 pixels from center horizontally
    horizontal = 1. / (20. * W / 128.) ** 2 * torch.pow(torch.abs(horizontal), 2)

    vertical = torch.swapaxes(torch.arange(H).repeat(B, C, W, 1), 3, 2).type_as(d) - H // 2

    # If height is 128, weight is 4 at 35 pixels from center vertically
    vertical = 1. / (35. * H / 128.) ** 2 * torch.pow(torch.abs(vertical), 2)
    return horizontal + vertical


def compute_inference_posterior(d, pixel_weights=False, lam=1, norm='tv'):
    """
    Compute MAP posterior from d
    If posterior_pixel_weights, quadratically increase pixel weight based on distance from center
    """
    weight = torch.ones_like(d)
    if pixel_weights:
        weight = _compute_increasing_weight(d)
    if norm == 'tv':
        return compute_total_variation_loss(d, lam, pixel_weight=weight)
    elif norm == 'l1':
        return lam * (weight * torch.abs(d)).sum(dim=(1, 2, 3)).mean()
    elif norm == 'l2':
        return lam * (weight * torch.square(d)).sum(dim=(1, 2, 3)).mean()
    elif norm == 'l4':
        return lam * torch.norm(weight * d, p=4)
    elif norm == 'nuc':
        return lam * torch.norm(d, p='nuc', dim=(2, 3)).sum()
    else:
        raise ValueError(f'Inference norm {norm} invalid')


def compute_total_variation_loss(x, lam, pixel_weight=False):
    """
    Compute average over batch of the TV norm of x
    """
    B = x.shape[0]
    tv_h = torch.pow(x[:, :, 1:, :-1] - x[:, :, :-1, :-1], 2)
    tv_w = torch.pow(x[:, :, :-1, 1:] - x[:, :, :-1, :-1], 2)
    tv_h_w = tv_h + tv_w
    tv_h_w[tv_h_w != 0] = torch.sqrt(tv_h_w[tv_h_w != 0])
    if pixel_weight is not False:
        tv_h_w = pixel_weight[:, :, 1:, 1:] * tv_h_w
    tv = tv_h_w.sum()
    return 1. / B * lam * tv


def _generate_mask(p_id, resize, orig_scale_fac):
    """
    Generate mask for p_id
    """
    transform = get_transforms(resize=resize, augment=False)
    frames = HeartEchoDataset([p_id],
                              label_type=LABELTYPE.VISIBLE_FAILURE_STRONG,
                              scale_factor=orig_scale_fac,
                              transform=transform,
                              frame_block_size=1,
                              procs=32)
    # Skip unavailable patients
    if len(frames) == 0:
        return

    # Extract mask from patient
    H, W = frames[0][0].shape[1:]
    vid = torch.cat(tuple(frame[0] for frame in frames), dim=0)
    mask = torch.std(vid, dim=0)
    mask[mask < 0.01] = 0

    # Set a border of 5 pixels on each side
    #     mask[:H//16,:] = 0
    #     mask[:,:W//16] = 0
    #     mask[-H//16:,:] = 0
    #     mask[:,-W//16:] = 0

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

    # Remove some handpicked pixels for certain patients
    #     if '70' in p_id:
    #         mask[int(0.9*H):,:] = 0

    # Augment mask with convex hull
    hull = ConvexHull(torch.nonzero(mask))
    hull_mask = torch.ones(mask.shape)
    for i in range(hull_mask.shape[0]):
        for j in range(hull_mask.shape[1]):
            point = np.array([i, j, 1])
            for eq in hull.equations:
                if eq.dot(point) > 0:
                    hull_mask[i, j] = 0

    #     # Compute arc corner points from hull and recompute it
    #     corners = np.int32(get_arc_points_from_mask(hull_mask))
    #     corners[corners[:,0]>=H,0] = H-1
    #     corners[corners[:,1]>=W,1] = W-1
    #     corners[corners[:,0]<0,0] = 0
    #     corners[corners[:,1]<0,1] = 0

    #     hull_mask[corners[:,0],corners[:,1]] = 1
    #     hull = ConvexHull(torch.nonzero(hull_mask))
    #     final_mask = torch.ones(mask.shape)
    #     for i in range(final_mask.shape[0]):
    #         for j in range(final_mask.shape[1]):
    #             point = np.array([i,j,1])
    #             for eq in hull.equations:
    #                 if eq.dot(point) > 0:
    #                     final_mask[i,j] = 0
    return p_id, hull_mask


def _compute_heart_mask(p_id, resize, orig_scale_fac):
    transform = get_transforms(resize=resize, augment=False)
    frames = HeartEchoDataset([p_id],
                              label_type=LABELTYPE.VISIBLE_FAILURE_STRONG,
                              scale_factor=orig_scale_fac,
                              transform=transform,
                              frame_block_size=1,
                              procs=32)
    # Skip unavailable patients
    if len(frames) == 0:
        return

    # Concatenate frames of video
    vid = torch.cat(tuple(frame[0] for frame in frames), dim=0)

    # Assemble crude mask by suppressing pixels with low variance
    mask = torch.std(vid, dim=0)
    mask[mask < torch.quantile(mask, 0.9)] = 0
    mask[mask != 0] = 1

    # Refine mask by computing convex hull
    hull = ConvexHull(torch.nonzero(mask))
    hull_mask = torch.ones(mask.shape)
    for i in range(hull_mask.shape[0]):
        for j in range(hull_mask.shape[1]):
            point = np.array([i, j, 1])
            for eq in hull.equations:
                if eq.dot(point) > 0:
                    hull_mask[i, j] = 0

    # Assemble heart mask by predicting ellipse from hull mask
    def is_valid(X, y):
        return np.unique(y).shape[0] > 1

    model = RANSACRegressor(
        LogisticRegression(),
        is_data_valid=is_valid,
        max_trials=10000,
        min_samples=50,
        residual_threshold=0.1
    )

    # Extract samples within and outside ellipse from convex hull
    indices = torch.cat((hull_mask.nonzero(), (hull_mask == 0).nonzero()))
    labels = torch.cat((torch.ones(hull_mask.nonzero().shape[0]), torch.zeros((hull_mask == 0).nonzero().shape[0])))

    # Center and square points for ellipse equation
    ransac_data = ((indices - 64) / 64).square()

    # Shuffle points for ransac
    perm = torch.randperm(ransac_data.shape[0])
    ransac_data = ransac_data[perm, :]
    ransac_labels = labels[perm]

    # Fit ransac with logisitc regression to extracted data
    model.fit(ransac_data, ransac_labels)

    # Assemble mask from fitted model
    preds = model.predict(ransac_data)
    est_mask = torch.zeros_like(mask)
    for (x, y), label in zip(indices[perm], preds):
        if label > 0:
            est_mask[x, y] = 1
    return p_id, est_mask


def generate_masks(resize, orig_scale_fac, heart_mask=False):
    mask_path = os.path.expanduser(os.path.join('~', '.echo-net', 'masks'))
    mask_fn = os.path.join(mask_path,
                           f'{resize}_{int(100 * orig_scale_fac)}_percent{"" if not heart_mask else f"_heart_mask"}.pt')
    # Get a mask for each patient
    print("Assembling echo masks.")
    masks = {}
    p_ids = TRAIN_PATIENT_IDS + VAL_PATIENT_IDS + TEST_PATIENT_IDS + MINOR_ANO_TEST_PATIENT_IDS_CV + PH_HEALTHY[
        'KAPAP'] + PH_ANOMALIES['KAPAP'] + PH_HEALTHY['KAAP'] + PH_ANOMALIES['KAAP'] + PH_HEALTHY['LA'] + PH_ANOMALIES[
                'LA']

    with mp.Pool(16) as pool:
        result = pool.starmap_async(_generate_mask if not heart_mask else _compute_heart_mask,
                                    zip(p_ids, repeat(resize), repeat(orig_scale_fac)),
                                    chunksize=1)
        num_chunks = result._number_left
        pbar = tqdm(total=num_chunks)

        mp_done = False
        curr_num_done = 0

        while mp_done is not True:
            time.sleep(1)

            num_done = num_chunks - result._number_left

            # Update progress
            curr_diff = num_done - curr_num_done

            if curr_diff != 0:
                pbar.update(curr_diff)
                curr_num_done = num_done

            if num_done == num_chunks:
                mp_done = True
        pbar.close()
    # Join results
    for tup in result.get():
        if tup is not None:
            p_id = tup[0]
            masks[p_id] = tup[1]

    #     for patient in tqdm(p_ids):
    #         p_id = f'{patient}CV'
    #         mask = _generate_mask(p_id, resize, orig_scale_fac)
    #         if mask is None:
    #             continue
    #         masks[p_id] = mask

    torch.save(masks, mask_fn)


def generate_mask_corners(orig_scale_fac):
    # Assemble precomputation paths
    mask_path = os.path.expanduser(os.path.join('~', '.echo-net', 'masks'))
    mask_fn = os.path.join(mask_path, f'{-1}_{int(100 * orig_scale_fac)}_percent.pt')
    corner_path = os.path.expanduser(os.path.join('~', '.echo-net', 'mask_corners'))
    corners_fn = os.path.join(corner_path, f'{int(100 * orig_scale_fac)}_percent.pt')

    # Load masks
    if not os.path.exists(mask_fn):
        utilities.generate_masks(-1, orig_scale_fac)
    masks = torch.load(mask_fn)

    # Generate Corners
    corners = {}
    for p_id in masks:
        corners[p_id] = np.int32(get_arc_points_from_mask(masks[p_id]))
    torch.save(corners, corners_fn)


def circle_from_points(a, b, c):
    """
    Compute circle radius and center from three points on it
    """
    c = Circle(a, b, c)
    return c.radius, np.array(c.center)


def _get_first_non_zero_horizontal(mask, outer_iter, inner_iter):
    """
    Search for non-zero pixel position,
    iterating over y coordinate with outer_iter in outer loop,
    and over x coordinate with inner_iter in inner loop
    """
    found = False
    outer_iter = list(outer_iter)
    inner_iter = list(inner_iter)
    counter = 0
    for i in outer_iter:
        if found:
            break
        for j in inner_iter:
            counter += 1
            if not found and mask[i, j] != 0:
                point = [i, j]
                found = True
                break
    return point


def _get_first_non_zero_vertical(mask, outer_iter, inner_iter):
    """
    Search for non-zero pixel position,
    iterating over y coordinate with outer_iter in outer loop,
    and over x coordinate with inner_iter in inner loop
    """
    found = False
    outer_iter = list(outer_iter)
    inner_iter = list(inner_iter)
    counter = 0
    for j in outer_iter:
        if found:
            break
        for i in inner_iter:
            counter += 1
            if not found and mask[i, j] != 0:
                point = [i, j]
                found = True
                break
    return point


def get_arc_points_from_mask(mask):
    # Get min and max nonzero entries
    mask_entries = torch.nonzero(mask)
    min_y, min_x = torch.min(mask_entries, dim=0).values
    max_y, max_x = torch.max(mask_entries, dim=0).values

    # Get top point
    top_candidates = mask_entries[mask_entries[:, 0] == min_y]
    top_left = torch.tensor([min_y, torch.min(top_candidates, dim=0).values[1]])
    top_right = torch.tensor([min_y, torch.max(top_candidates, dim=0).values[1]])

    left_candidates = mask_entries[mask_entries[:, 1] == min_x]
    left_top = torch.tensor([torch.min(left_candidates, dim=0).values[0], min_x])
    left_bot = torch.tensor([torch.max(left_candidates, dim=0).values[0], min_x])

    right_candidates = mask_entries[mask_entries[:, 1] == max_x]
    right_top = torch.tensor([torch.min(right_candidates, dim=0).values[0], max_x])
    right_bot = torch.tensor([torch.max(right_candidates, dim=0).values[0], max_x])

    left_line = Line(top_left, left_top)
    right_line = Line(top_right, right_top)
    top = np.array(left_line.intersection(right_line)[0])

    # Compute bottom point
    bot = []
    bot_candidates = mask_entries[mask_entries[:, 0] == max_y]
    bot.append(torch.tensor([[max_y, torch.min(bot_candidates, dim=0).values[1]]]))
    bot.append(torch.tensor([[max_y, torch.max(bot_candidates, dim=0).values[1]]]))
    bot = np.concatenate(bot)
    bot = 0.5 * (bot[0] + bot[1])

    # Create arc circle
    arc = Circle(left_bot, bot, right_bot)

    # Compute left and right points
    left_int = arc.intersection(left_line)
    right_int = arc.intersection(right_line)

    # Always take point with highest first coordinate (lowest)
    if len(left_int) == 1 or left_int[0][0] > left_int[1][0]:
        left = left_int[0]
    else:
        left = left_int[1]
    if len(right_int) == 1 or right_int[0][0] > right_int[1][0]:
        right = right_int[0]
    else:
        right = right_int[1]
    left = np.array(left)
    right = np.array(right)

    corners = np.array([top, right, bot, left])

    return corners