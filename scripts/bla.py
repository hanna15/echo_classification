import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils.transforms2 import get_transforms
from echo_ph.data import EchoDataset
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from echo_ph.models.resnets import get_resnet18
from echo_ph.models.resnet_3d import get_resnet3d_18
from echo_ph.visual.video_saver import VideoSaver
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from utils.helpers import get_index_file_path
import os

"""
A script to create grad-cam visualisations on spatial models, either saving as frames or videos.
Can let it run on all frames in an entire dataset, or specified videos.
"""

parser = ArgumentParser(
    description='Arguments for visualising grad cam',
    formatter_class=ArgumentDefaultsHelpFormatter)
# Arguments per model, to be averaged
parser.add_argument('--model_dir_paths', nargs='+',
                    help='set to path of a model state dict, to evaluate on. If None, use resnet18 pretrained on '
                         'Imagenet only.')
parser.add_argument('--model_types', default=['temporal', 'temporal'], nargs='+',
                    help='set model types (spatial / temporal) for each of the provided model paths (same order)')
parser.add_argument('--views',  nargs='+', default=['KAPAP', 'CV'])
parser.add_argument('--size', default=224, type=int, help='Size of images (frames) to resize to')
# Arguments that must be the same for all models
parser.add_argument('--label_type', default='2class_drop_ambiguous',
                    choices=['2class', '2class_drop_ambiguous', '3class'])
parser.add_argument('--cache_dir', default='~/.heart_echo')
parser.add_argument('--scale', default=0.25)
parser.add_argument('--k', default=10, type=int, help='Set total number of folds')
parser.add_argument('--n_workers', default=8, type=int)
parser.add_argument('--max_p', default=95, type=int)
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('--clip_len', type=int, default=0, help='How many frames to select per video')
parser.add_argument('--period', type=int, default=1, help='Sample period, sample every n-th frame')
# Optional additional arguments
parser.add_argument('--min_expansion', action='store_true',
                    help='Percentile for min expansion frames instead of maximum')
parser.add_argument('--num_rand_frames', type=int, default=None,
                    help='Set this only if get random frames instead of max/min')
parser.add_argument('--train_set', action='store_true', help='Also evaluate on the training set')
parser.add_argument('--segm_only', action='store_true', help='Only evaluate on the segmentation masks')
parser.add_argument('--video_ids', default=None, nargs='+', type=int, help='Get results for specific video ids')
parser.add_argument('--crop', action='store_true', help='set this flag to crop to corners')


def get_data_loader(fold, view='KAPAP', train=False, temp=False):
    if args.video_ids is None:
        index_file_path = get_index_file_path(args.k, fold, args.label_type, train=train)
    else:
        index_file_path = None
    label_path = os.path.join('label_files', 'labels_' + args.label_type + '.pkl')
    aug_type = 5 if train else 0
    transforms = get_transforms(index_file_path, dataset_orig_img_scale=args.scale, resize=args.size,
                                augment=aug_type, fold=fold, valid=(not train), view=view,
                                crop_to_corner=args.crop, segm_mask_only=args.segm_only)
    dataset = EchoDataset(index_file_path, label_path, cache_dir=args.cache_dir,
                          transform=transforms, scaling_factor=args.scale, procs=args.n_workers,
                          percentile=args.max_p, view=view, min_expansion=args.min_expansion,
                          num_rand_frames=args.num_rand_frames, segm_masks=args.segm_only, video_ids=args.video_ids,
                          temporal=temp, clip_len=args.clip_len, period=args.period)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    return data_loader


def foo(data_loader, model, device, temporal=False):
    outputs = []
    targets = []
    samples = []
    for batch in data_loader:
        img = batch['frame'].to(device)
        if temporal:
            img = img.transpose(2, 1)  # Reshape to: (batch_size, channels, seq-len, W, H)
        sample_name = batch['sample_name']
        target = batch['label']
        out = model(img)
        outputs.extend(out.cpu().detach().numpy())
        targets.extend(target)
        samples.extend(sample_name)
    return outputs, targets, samples


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold in range(args.k):
        num_classes = 2 if args.label_type.startswith('2') else 3
        total_outs = []
        total_targets = []
        total_samples = []
        for model_dir, model_type, view in zip(args.model_dir_paths, args.model_types, args.views):
            if model_type == 'temporal' or model_type == 'temp':
                temp = True
                model = get_resnet3d_18(num_classes=num_classes, model_type='r3d_18').to(device)
            else:  # spatial
                temp = False
                model = get_resnet18(num_classes=num_classes).to(device)
            model_path = sorted(os.listdir(model_dir))[fold]  # fetch the model corresponding to corresponding fold
            model_path = os.path.join(model_dir, model_path)
            val_data_loader = get_data_loader(fold, view, temp=temp)
            if args.train_set:
                train_data_loader = get_data_loader(fold, view, train=True, temp=temp)
            print(f'Done loading data for fold {fold}, for model {model_path}')
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            model = model.to(device)
            outputs, targets, samples = foo(val_data_loader, model, device, temporal=temp)
            total_outs.append(outputs)
            total_targets.append(targets)
            total_samples.append(samples)


if __name__ == '__main__':
    args = parser.parse_args()
    main()
