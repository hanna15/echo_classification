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
parser.add_argument('--num_rand_frames', type=int, default=None,
                    help='Set this only if get random frames instead of max/min')
parser.add_argument('--crop', action='store_true', help='If crop to corners')
parser.add_argument('--save', action='store_true', help='If to save grad cam images')
parser.add_argument('--show', action='store_true', help='If to show grad cam images')
parser.add_argument('--train_set', action='store_true', help='Also get grad cam for the images in the training set, '
                                                             'with random augmentation (type 3)')


def get_data_loader(train=False):
    idx_dir = 'index_files' if args.k is None else os.path.join('index_files', 'k' + str(args.k))
    idx_file_end = '' if args.fold is None else '_' + str(args.fold)
    idx_file_base_name = 'train_samples_' if train else 'valid_samples_'
    index_file_path = os.path.join(idx_dir, idx_file_base_name + args.label_type + idx_file_end + '.npy')
    label_path = os.path.join('label_files', 'labels_' + args.label_type + '.pkl')
    aug_type = 3 if train else 0
    transforms = get_transforms(index_file_path, dataset_orig_img_scale=args.scale, resize=224,
                                augment=aug_type, fold=args.fold, valid=(not train), view=args.view, crop_to_corner=args.crop)
    valid_dataset = EchoDataset(index_file_path, label_path, cache_dir=args.cache_dir,
                                transform=transforms, scaling_factor=args.scale, procs=args.n_workers,
                                percentile=args.max_p, view=args.view, min_expansion=args.min_expansion)
    data_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=args.n_workers)
    return data_loader


def get_save_grad_cam_images(data_loader, model, cam, device, subset='valid'):
    target_category = None
    if args.save:
        model_name = os.path.basename(args.model_path)[:-3]
        output_dir = os.path.join('grad_cam_vis', model_name, subset)
        os.makedirs(output_dir, exist_ok=True)
    for batch in data_loader:
        img = batch['frame'].to(device)
        sample_name = batch['sample_name'][0]
        label = batch['label'][0].item()
        pred = torch.max(model(img), dim=1).indices[0].item()
        corr = 'CORR' if label == pred else 'WRONG'
        title = f'{sample_name}-{corr}-{label}.jpg'
        grayscale_cam = cam(input_tensor=img,
                            target_category=target_category)
        img = np.stack((img.squeeze().cpu(),) * 3, axis=-1)  # create a 3-channel image from the grayscale img
        cam_image = show_cam_on_image(img, grayscale_cam[0])
        if args.show:
            plt.imshow(cam_image)
            plt.title(title)
            plt.show()
        if args.save:
            cv2.imwrite(os.path.join(output_dir, title), cam_image)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_data_loader = get_data_loader()
    if args.train_set:
        train_data_loader = get_data_loader(train=True)
    print("Done loading data")
    num_classes = 2 if args.label_type.startswith('2') else 3
    model = get_resnet18(num_classes=num_classes)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path))
    model.eval()
    model = model.to(device)

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    print("Done initialising grad cam with model")

    get_save_grad_cam_images(val_data_loader, model, cam, device)
    if args.train_set:
        get_save_grad_cam_images(train_data_loader, model, cam, device, subset='train')


if __name__ == '__main__':
    args = parser.parse_args()
    main()