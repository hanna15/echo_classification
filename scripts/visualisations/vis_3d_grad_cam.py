import os
import torch
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from medcam import medcam
from utils.transforms2 import get_transforms
from echo_ph.data import EchoDataset
from echo_ph.models.resnet_3d import get_resnet3d_18, get_resnet3d_50

parser = ArgumentParser(
    description='Arguments for visualising 3D grad cam',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_path', default=None, help='set to path of a model state dict, to evaluate on. '
                                                       'If None, use resnet18 pretrained on Imagenet only.')
parser.add_argument('--model', default='r3d_50', choices=['r2plus1d_18', 'mc3_18', 'r3d_18', 'r3d_50'],
                    help='What model architecture to use.')
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
# Optional additional arguments
parser.add_argument('--num_rand_frames', type=int, default=None,
                    help='Set this only if get random frames instead of max/min')
parser.add_argument('--crop', action='store_true', help='If crop to corners')
parser.add_argument('--train_set', action='store_true', help='Also get grad cam for the images in the training set, '
                                                             'with random augmentation (type 3)')

# Temporal param
parser.add_argument('--clip_len', type=int, default=12, help='How many frames to select per video')
parser.add_argument('--period', type=int, default=1, help='Sample period, sample every n-th frame')
parser.add_argument('--zip', action='store_true', help='Zip the resulting dir')


def get_data_loader(train=False):
    idx_dir = 'index_files' if args.k is None else os.path.join('index_files', 'k' + str(args.k))
    idx_file_base_name = 'train_samples_' if train else 'valid_samples_'
    idx_file_end = '' if args.fold is None else '_' + str(args.fold)
    index_file_path = os.path.join(idx_dir, idx_file_base_name + args.label_type + idx_file_end + '.npy')
    label_path = os.path.join('label_files', 'labels_' + args.label_type + '.pkl')
    aug_type = 4 if train else 0
    transforms = get_transforms(index_file_path, dataset_orig_img_scale=args.scale, resize=224,
                                augment=aug_type, fold=args.fold, valid=(not train), view=args.view,
                                crop_to_corner=args.crop)
    dataset = EchoDataset(index_file_path, label_path, cache_dir=args.cache_dir,
                          transform=transforms, scaling_factor=args.scale, procs=args.n_workers,
                          view=args.view,  num_rand_frames=args.num_rand_frames, temporal=True,
                          clip_len=args.clip_len, period=args.period)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.n_workers)
    return data_loader


def get_save_grad_cam_images(data_loader, model, device, subset='valid'):
    base_res_dir = os.path.join('3d_plotsX', subset)
    os.makedirs(base_res_dir, exist_ok=True)
    for batch in data_loader:
        inp = batch['frame'].to(device)
        inp = inp.transpose(2, 1)  # Reshape to: (batch_size, channels, seq-len, W, H)
        sample_name = batch['sample_name'][0]
        label = batch['label'][0].item()
        out, att = model(inp)
        pred = torch.max(out, dim=1).indices[0].item()
        corr = 'CORR' if label == pred else 'WRONG'
        video = np.swapaxes(inp, 1, 2)[0]
        att_video = np.swapaxes(att, 1, 2)[0]
        print('len video', len(video))
        sample_dir = os.path.join(base_res_dir, sample_name)
        os.makedirs(sample_dir, exist_ok=True)
        for i in range(len(video)):
            img = video[i]
            att_img = att_video[i]
            plt.imshow(img.squeeze().cpu(), cmap='Greys_r')
            plt.imshow(att_img.squeeze().cpu(), cmap='jet', alpha=0.5)
            title = f'{sample_name}-{i}-{corr}-{label}.jpg'
            plt.title(title)
            plt.savefig(os.path.join(sample_dir, 'frame_' + str(i) + '.png'))
    if args.zip:
        os.system(f'zip -r {base_res_dir}.zip {base_res_dir}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_data_loader = get_data_loader()
    print("Done loading valid data")
    num_classes = 2 if args.label_type.startswith('2') else 3
    if args.model.endswith('18'):
        model = get_resnet3d_18(num_classes=num_classes, model_type=args.model).to(device)
    else:
        model = get_resnet3d_50(num_classes=num_classes).to(device)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = medcam.inject(model, return_attention=True, backend='gcam')
    model.eval()
    get_save_grad_cam_images(val_data_loader, model, device, subset='valid')
    if args.train_set:
        train_data_loader = get_data_loader(train=True)
        get_save_grad_cam_images(train_data_loader, model, device, subset='train')


if __name__ == '__main__':
    args = parser.parse_args()
    main()