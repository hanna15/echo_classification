import torch
from torch import nn
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils.transforms2 import get_transforms
from echo_ph.data import EchoDataset
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from echo_ph.models.resnets import get_resnet18
from echo_ph.visual.video_saver import VideoSaver
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
parser.add_argument('--all_frames', action='store_true', default=None,
                    help='Get all frames of a video')
parser.add_argument('--crop', action='store_true', help='If crop to corners')
parser.add_argument('--save', action='store_true', help='If to save grad cam images')
parser.add_argument('--show', action='store_true', help='If to show grad cam images')
parser.add_argument('--train_set', action='store_true', help='Also get grad cam for the images in the training set, '
                                                             'with random augmentation (type 3)')
parser.add_argument('--segm_only', action='store_true', help='Only evaluate on the segmentation masks')
parser.add_argument('--video_ids', default=None, nargs='+', type=int, help='Instead of getting results acc.to index file, '
                                                                 'get results for specific video ids')
parser.add_argument('--save_video', action='store_true', help='If also to save video visualisations')


def get_data_loader(train=False):
    if args.video_ids:
        index_file_path = None
    else:
        idx_dir = 'index_files' if args.k is None else os.path.join('index_files', 'k' + str(args.k))
        idx_file_end = '' if args.fold is None else '_' + str(args.fold)
        idx_file_base_name = 'train_samples_' if train else 'valid_samples_'
        index_file_path = os.path.join(idx_dir, idx_file_base_name + args.label_type + idx_file_end + '.npy')
    label_path = os.path.join('label_files', 'labels_' + args.label_type + '.pkl')
    aug_type = 3 if train else 0
    transforms = get_transforms(index_file_path, dataset_orig_img_scale=args.scale, resize=224,
                                augment=aug_type, fold=args.fold, valid=(not train), view=args.view,
                                crop_to_corner=args.crop, segm_mask_only=args.segm_only)
    dataset = EchoDataset(index_file_path, label_path, cache_dir=args.cache_dir,
                          transform=transforms, scaling_factor=args.scale, procs=args.n_workers,
                          percentile=args.max_p, view=args.view, min_expansion=args.min_expansion,
                          num_rand_frames=args.num_rand_frames, segm_masks=args.segm_only, video_ids=args.video_ids,
                          all_frames=args.all_frames)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.n_workers)
    return data_loader


def get_save_grad_cam_images(data_loader, model, cam, device, subset='valid'):
    target_category = None
    if args.save:
        model_name = os.path.basename(args.model_path)[:-3]
        output_dir = os.path.join('grad_cam_vis', model_name, subset)
        os.makedirs(output_dir, exist_ok=True)
    for batch in data_loader:
        all_frames = []
        img = batch['frame'].to(device)
        sample_name = batch['sample_name'][0]
        video_id = int(sample_name.split('_')[0])
        label = batch['label'][0].item()
        pred = torch.max(model(img), dim=1).indices[0].item()
        corr = 'CORR' if label == pred else 'WRONG'
        title = f'{sample_name}-{corr}-{label}.jpg'
        out_dir = os.path.join(output_dir,  f'{sample_name}_{corr}_{label}')
        os.makedirs(out_dir, exist_ok=True)
        grayscale_cam = cam(input_tensor=img,
                            target_category=target_category)
        img = np.stack((img.squeeze().cpu(),) * 3, axis=-1)  # create a 3-channel image from the grayscale img
        try:
            cam_image = show_cam_on_image(img, grayscale_cam[0])
            all_frames.append(cam_image)
            if args.show:
                plt.imshow(cam_image)
                plt.title(title)
                plt.show()
            if args.save:
                cv2.imwrite(os.path.join(out_dir, title), cam_image)
            if args.save_video:
                vs = VideoSaver(video_id, all_frames)  # Use default fps and max_frames
                vs.save_video()
                # size = (all_frames[0].shape[0], all_frames[0].shape[1])
                # video_dir = 'vis_videos'
                # os.makedirs(video_dir, exist_ok=True)
                # result = cv2.VideoWriter(os.path.joinvideo_dir, f'{sample_name}_{corr}_{label}.avi',
                #                          cv2.VideoWriter_fourcc(*'MJPG'), 24, size)
                # for frame in all_frames[0:]:
                #     result.write(frame)
                # result.release()
                # cv2.destroyAllWindows()
        except:
            print(f'failed for sample {sample_name}, max is {img.max()}, min is {img.min()}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_data_loader = get_data_loader()
    if args.train_set:
        train_data_loader = get_data_loader(train=True)
    print("Done loading data")
    num_classes = 2 if args.label_type.startswith('2') else 3
    model = get_resnet18(num_classes=num_classes)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
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