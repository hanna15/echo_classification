import torch
from torch import nn
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils.transforms2 import get_transforms
from echo_ph.data import EchoDataset
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from echo_ph.models.resnets import get_resnet18
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import os

parser = ArgumentParser(
    description='Arguments for visualising grad cam',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_path', default=None, help='set to path of a model state dict, to evaluate on. '
                                                       'If None, use resnet18 pretrained on Imagenet only.')
parser.add_argument('--label_type', default='2class_drop_ambiguous',
                    choices=['2class', '2class_drop_ambiguous', '3class'])
parser.add_argument('--cache_dir', default='~/.heart_echo')
parser.add_argument('--scale', default=0.25)
parser.add_argument('--view', default='KAPAP')
parser.add_argument('--fold', default=0, type=int,
                    help='In case of k-fold cross-validation, set the current fold for this training.'
                         'Will be used to fetch the relevant k-th train and valid index file')
parser.add_argument('--k', default=10, type=int,
                    help='In case of k-fold cross-validation, set the k, i.e. how many folds all in all. '
                         'Will be used to fetch the relevant k-th train and valid index file')
parser.add_argument('--n_workers', default=8, type=int)
parser.add_argument('--max_p', default=95, type=int)
# Optional additional arguments
parser.add_argument('--min_expansion', action='store_true',
                    help='Percentile for min expansion frames instead of maximum')
parser.add_argument('--use_cuda', action='store_true')
parser.add_argument('--crop', action='store_true', help='If crop to corners')
parser.add_argument('--save', action='store_true', help='If to save grad cam images')
parser.add_argument('--show', action='store_true', help='If to show grad cam images')


def get_data_loader():
    idx_dir = 'index_files' if args.k is None else os.path.join('index_files', 'k' + str(args.k))
    idx_file_end = '' if args.fold is None else '_' + str(args.fold)

    valid_index_file_path = os.path.join(idx_dir, 'valid_samples_' + args.label_type + idx_file_end + '.npy')
    label_path = os.path.join('label_files', 'labels_' + args.label_type + '.pkl')

    valid_transforms = get_transforms(valid_index_file_path, dataset_orig_img_scale=args.scale, resize=224,
                                      augment=0, fold=args.fold, valid=True, view=args.view, crop_to_corner=args.crop)
    valid_dataset = EchoDataset(valid_index_file_path, label_path, cache_dir=args.cache_dir,
                                transform=valid_transforms, scaling_factor=args.scale, procs=args.n_workers,
                                percentile=args.max_p, view=args.view, min_expansion=args.min_expansion)

    data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.n_workers)
    return data_loader


def main():
    data_loader = get_data_loader()
    print("Done loading data")
    num_classes = 2 if args.label_type.startswith('2') else 3
    model = get_resnet18(num_classes=num_classes)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))
    model.eval()
    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda)
    print("Done initialising grad cam with model")
    target_category = None
    if args.save:
        output_dir = 'grad_cam_vis'
        os.makedirs(output_dir, exist_ok=True)
    for batch in data_loader:
        img = batch['frame']
        sample_name = batch['sample_name'][0]
        label = batch['label'][0].item()
        pred = torch.max(model(img), dim=1).indices[0].item()
        corr = 'CORR' if label == pred else 'WRONG'
        grayscale_cam = cam(input_tensor=img,
                            target_category=target_category)
        img = np.stack((img.squeeze(),)*3, axis=-1)  # create a 3-channel image from the grayscale img
        cam_image = show_cam_on_image(img, grayscale_cam[0])
        if args.show:
            plt.imshow(cam_image)
            plt.show()
        if args.save:
            cv2.imwrite(os.path.join(output_dir, f'{sample_name}-{corr}-{label}.jpg'), cam_image)


if __name__ == '__main__':
    args = parser.parse_args()
    main()