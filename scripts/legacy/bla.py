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
A script for multi-view classification
"""

parser = ArgumentParser(
    description='Arguments for multi-view classification',
    formatter_class=ArgumentDefaultsHelpFormatter)
# Arguments per model, to be averaged
parser.add_argument('--model_dir_paths', nargs='+',
                    help='set to path of a model state dict, to evaluate on. If None, use resnet18 pretrained on '
                         'Imagenet only.')
parser.add_argument('--views',  nargs='+', default=['KAPAP', 'CV'])
parser.add_argument('--weights',  nargs='+', default=[0.7, 0.3])
# Arguments that must be the same for all models
parser.add_argument('--size', default=224, type=int, help='Size of images (frames) to resize to')
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
parser.add_argument('--temp', action='store_true', help='set this flag if temporal model')


def get_data_loader(fold, views=['KAPAP', 'CV'], train=False, temp=False):
    if args.video_ids is None:
        index_file_path = get_index_file_path(args.k, fold, args.label_type, train=train)
    else:
        index_file_path = None
    label_path = os.path.join('label_files', 'labels_' + args.label_type + '.pkl')
    aug_type = 5 if train else 0
    transforms = get_transforms(index_file_path, dataset_orig_img_scale=args.scale, resize=args.size,
                                augment=aug_type, fold=fold, valid=(not train), view=views,
                                crop_to_corner=args.crop, segm_mask_only=args.segm_only)
    dataset = EchoDataset(index_file_path, label_path, cache_dir=args.cache_dir,
                          transform=transforms, scaling_factor=args.scale, procs=args.n_workers,
                          percentile=args.max_p, view=views, min_expansion=args.min_expansion,
                          num_rand_frames=args.num_rand_frames, segm_masks=args.segm_only, video_ids=args.video_ids,
                          temporal=temp, clip_len=args.clip_len, period=args.period)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)
    return data_loader


def read(data_loader, model, device, temporal=False, view='KAPAP'):
    outputs = []
    targets = []
    samples = []
    for batch in data_loader:
        img = batch['frame'][view].to(device)  # Select frames for correct view
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
    res_dir = 'results_averaged' + str([view for view in args.views])
    for fold in range(args.k):
        num_classes = 2 if args.label_type.startswith('2') else 3
        total_outs = []
        total_targets = []
        total_samples = []
        total_vid_ids = []
        val_data_loader = get_data_loader(fold, args.views, temp=args.temp)
        if args.train_set:
            train_data_loader = get_data_loader(fold, args.views, train=True, temp=args.temp)
        for model_dir, view in zip(args.model_dir_paths, args.views):
            if args.temp:
                model = get_resnet3d_18(num_classes=num_classes, model_type='r3d_18').to(device)
            else:  # spatial
                model = get_resnet18(num_classes=num_classes).to(device)
            model_path = sorted(os.listdir(model_dir))[fold]  # fetch the model corresponding to corresponding fold
            model_path = os.path.join(model_dir, model_path)
            print(f'Done loading data for fold {fold}, for model {model_path}')
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            model = model.to(device)
            outputs, targets, samples = read(val_data_loader, model, device, temporal=args.temp, view=view)
            total_outs.append(outputs)
            total_targets.append(targets)
            total_samples.append(samples)
            vid_ids = [s.split('_')[0] for s in samples]
            total_vid_ids.append(vid_ids)
        outs = np.average(np.asarray(total_outs), axis=0, weights=args.weights)
        np.save(os.path.join(res_dir, 'fold' + str(fold) + '_val_preds.npy'), outs)
        np.save(os.path.join(res_dir, 'fold' + str(fold) + 'val_targets.npy'), total_targets[0])
        np.save(os.path.join(res_dir, 'fold' + str(fold) + 'val_samples.npy'), total_samples[0])
        print(set(total_vid_ids[0]) == set(total_vid_ids[1]))
        print(set(total_samples[0]) == set(total_samples[1]))
        print(total_samples[0] == total_samples[1])


if __name__ == '__main__':
    args = parser.parse_args()
    main()