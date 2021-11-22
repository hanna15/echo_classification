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
parser.add_argument('--max_frame', type=int, default=50,
                    help='Only valid in combination with all_frames flag. '
                         'Get sequential frames of a video from frame 0, but limit len to max_frame')
parser.add_argument('--crop', action='store_true', help='If crop to corners')
parser.add_argument('--save_frames', action='store_true', help='If to save grad cam images on a frame-level')
parser.add_argument('--save_video', action='store_true', help='If also to save video visualisations')
parser.add_argument('--show', action='store_true', help='If to show grad cam images')
parser.add_argument('--train_set', action='store_true', help='Also get grad cam for the images in the training set, '
                                                             'with random augmentation (type 3)')
parser.add_argument('--segm_only', action='store_true', help='Only evaluate on the segmentation masks')
parser.add_argument('--video_ids', default=None, nargs='+', type=int, help='Instead of getting results acc.to index file, '
                                                                 'get results for specific video ids')
parser.add_argument('--dynamic', action='store_true', help='If run on dynamic dataset')


def get_data_loader(video_ids, train=False):
    if args.dynamic:
        label_path = os.path.join('label_files', 'dynamic_test_labels.pkl')
    else:
        label_path = os.path.join('label_files', 'labels_' + args.label_type + '.pkl')
    aug_type = 5 if train else 0
    dyn = False
    index_path = None
    # Note: Ok to have index_path None in transforms, wout providing video_ids, because only need index_path
    # when creating masks for first time => here, as we are in validation mode, we assume masks have already
    # been created
    transforms = get_transforms(index_path, dataset_orig_img_scale=args.scale, resize=224,
                                augment=aug_type, fold=args.fold, valid=(not train), view=args.view,
                                crop_to_corner=args.crop, segm_mask_only=args.segm_only)
    dataset = EchoDataset(index_path, label_path, video_ids=video_ids,
                          cache_dir=args.cache_dir, transform=transforms, scaling_factor=args.scale,
                          procs=args.n_workers, percentile=args.max_p, view=args.view, min_expansion=args.min_expansion,
                          num_rand_frames=args.num_rand_frames, segm_masks=args.segm_only,
                          all_frames=args.all_frames, max_frame=args.max_frame, dynamic=dyn)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.n_workers)
    return data_loader


def get_save_grad_cam_images(data_loader, model, cam, device, subset='valid'):
    target_category = None
    if args.save_frames or args.save_video:
        model_name = os.path.basename(args.model_path)[:-3]
        output_dir = os.path.join('grad_cam_vis', model_name, subset)
        os.makedirs(output_dir, exist_ok=True)
    video_cam_frames = []
    frame_titles = []
    first = True
    for batch in data_loader:
        img = batch['frame'].to(device)
        sample_name = batch['sample_name'][0]  # get first, because batch size 1
        video_id = sample_name.split('_')[0]  # get first, because batch size 1
        if first:
            print('Processing video', video_id)
            first = False
        label = batch['label'][0].item()  # get first, because batch size 1
        pred = torch.max(model(img), dim=1).indices[0].item()
        corr = 'CORR' if label == pred else 'WRONG'
        title = f'{sample_name}-{corr}-{label}.jpg'
        grayscale_cam = cam(input_tensor=img,
                            target_category=target_category)
        img = np.stack((img.squeeze().cpu(),) * 3, axis=-1)  # create a 3-channel image from the grayscale img
        try:
            cam_image = show_cam_on_image(img, grayscale_cam[0])
        except:
            print(f'failed for sample {sample_name}, max is {img.max()}, min is {img.min()}')
        video_cam_frames.append(cam_image)
        frame_titles.append(title)
        if args.show:
            plt.imshow(cam_image)
            plt.title(title)
            plt.show()
        if args.save_frames:
            out_dir = os.path.join(output_dir, str(video_id))
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, title), cam_image)
    if args.save_video:
        frame_corrs = np.asarray([frame_title.split('-')[1] for frame_title in frame_titles])
        ratio_corr = (frame_corrs == 'CORR').sum() / len(frame_titles)
        print('ratio corr', ratio_corr)
        if ratio_corr > 0.91 or ratio_corr < 0.3:  # Save only best and worst videos
            true_label = frame_titles[0].split('-')[-1][:-4]
            video_title = f'{video_id}-{ratio_corr:.2f}-{true_label}.jpg'
            out_dir = output_dir + '_video'
            vs = VideoSaver(video_title, video_cam_frames, out_dir=out_dir, fps=10)
            vs.save_video()


def get_video_ids(train=False):
    if args.dynamic: # Fetch the videos in the EchoNet-Dynamics validation dataset
        index_file_path = os.path.join('index_files', 'dynamic_test_index.npy')
        video_ids = np.load(index_file_path)
    elif args.video_ids: # Set video ids to the provided video ids
        video_ids = args.video_ids
    else:  # Fetch the appropriate index file
        idx_dir = 'index_files' if args.k is None else os.path.join('index_files', 'k' + str(args.k))
        idx_file_end = '' if args.fold is None else '_' + str(args.fold)
        idx_file_base_name = 'train_samples_' if train else 'valid_samples_'
        index_file_path = os.path.join(idx_dir, idx_file_base_name + args.label_type + idx_file_end + '.npy')
        video_ids = np.load(index_file_path)
    return video_ids


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Get video ids, because here we will have a seperate data loader for each video
    val_video_ids = get_video_ids()
    if args.train_set:
        train_video_ids = get_video_ids(train=True)
    # Get model
    num_classes = 2 if args.label_type.startswith('2') else 3
    model = get_resnet18(num_classes=num_classes)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    model = model.to(device)
    # Get Grad-CAM
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
    print("Done initialising grad cam with model")
    # Start to predict for each video_id
    for video_id in val_video_ids:
        val_data_loader = get_data_loader(video_ids=[video_id])
        get_save_grad_cam_images(val_data_loader, model, cam, device)
    if args.train_set:
        for video_id in train_video_ids:
            train_data_loader = get_data_loader(video_ids=[video_id], train=True)
            get_save_grad_cam_images(train_data_loader, model, cam, device, subset='train')


if __name__ == '__main__':
    args = parser.parse_args()
    main()
