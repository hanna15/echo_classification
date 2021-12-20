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
from echo_ph.visual.video_saver import VideoSaver
import cv2
import matplotlib.cm as cm

"""
A script to create grad-cam visualisations on temporal models, either saving as frames or videos.
Can let it run on all frames in an entire dataset, or specified videos.
"""


parser = ArgumentParser(
    description='Arguments for visualising 3D grad cam',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_path', default=None, help='set to path of a model state dict, to evaluate on. '
                                                       'If None, use resnet18 pretrained on Imagenet only.')
parser.add_argument('--out_dir', default='grad_cam_3d', help='Name of directory storing the results')
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

parser.add_argument('--video_ids', default=None, nargs='+', type=int,
                    help='Instead of getting results acc.to index file, get results for specific video ids')
parser.add_argument('--zip', action='store_true', help='Zip the resulting dir')
parser.add_argument('--save_video_clips', action='store_true', help='Save individual video clips')
parser.add_argument('--save_video', action='store_true', help='Save entire video, batching up video clips')
parser.add_argument('--save_frames', action='store_true', help='Save individual frames')
parser.add_argument('--all_frames', action='store_true', default=None,
                    help='Get all frames of a video')
parser.add_argument('--max_frame', type=int, default=50,
                    help='Only valid in combination with all_frames flag. '
                         'Get sequential frames of a video from frame 0, but limit len to max_frame.')
parser.add_argument('--show', action='store_true', help='If to show grad cam images')
# Temporal param
parser.add_argument('--clip_len', type=int, default=12, help='How many frames to select per video')
parser.add_argument('--period', type=int, default=1, help='Sample period, sample every n-th frame')


def get_data_loader(train=False):
    if args.video_ids:
        index_file_path = None
    else:
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
                          clip_len=args.clip_len, period=args.period, video_ids=args.video_ids,
                          all_frames=args.all_frames, max_frame=args.max_frame)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.n_workers)
    return data_loader


def normalize(x):
    """Normalizes data both numpy or tensor data to range [0,1]."""
    if isinstance(x, torch.Tensor):
        if torch.min(x) == torch.max(x):
            return torch.zeros(x.shape)
        return (x-torch.min(x))/(torch.max(x)-torch.min(x))
    else:
        if np.min(x) == np.max(x):
            return np.zeros(x.shape)
        return (x - np.min(x)) / (np.max(x) - np.min(x))


def overlay(raw_input, attention_map):
    """
    Overlay attention map on top of a raw image.
    :param raw_input: The original input frame (resized or original size). Shape should be ch x w x h.
    :param attention_map: The attention map of that frame. Shape should be ch x w x h.
    :return: The overlayed image of shape (W X H X 3) => needed to save as image.
    """
    attention_map = np.asarray(attention_map)
    raw_input = np.asarray(raw_input)
    raw_input = raw_input.transpose(1, 2, 0)  # Reshape to : W X H X CH
    attention_map = normalize(attention_map.astype(np.float))
    if np.max(raw_input) > 1:
        raw_input = raw_input.astype(float)
        raw_input /= 255
    attention_map = cm.jet_r(attention_map)[..., :3].squeeze()
    attention_map = (attention_map.astype(float) + raw_input.astype(float)) / 2
    attention_map *= 255
    return attention_map.astype(np.uint8)


def get_save_grad_cam_images(data_loader, model, device, subset='valid'):
    base_res_dir = os.path.join(args.out_dir, subset)
    os.makedirs(base_res_dir, exist_ok=True)
    video_clips = {}
    output_dir = os.path.join('grad_cam_vis', 'temp', subset)
    os.makedirs(output_dir, exist_ok=True)
    for batch in data_loader:
        inp = batch['frame'].to(device).transpose(2, 1)  # Reshape to: (batch_size, channels, seq-len, W, H)
        sample_name = batch['sample_name'][0]
        video_id = int(sample_name.split('_')[0])
        label = batch['label'][0].item()
        # Using model to predict, and get also grad-cam attention map.
        out, att = model(inp)
        pred = torch.max(out, dim=1).indices[0].item()
        corr = 'CORR' if label == pred else 'WRONG'
        title = f'{sample_name}-{corr}-{label}.jpg'
        video = np.swapaxes(inp, 1, 2)[0]  # re-shape to no_frames, ch, W, H
        att_video_clip = np.swapaxes(att, 1, 2)[0]  # re-shape to no_frames, ch, W, H
        print('len video', len(video))
        sample_dir = os.path.join(base_res_dir, sample_name)
        os.makedirs(sample_dir, exist_ok=True)
        att_clip = att_video_clip.squeeze(1).cpu().detach().numpy()
        raw_vid_clip = video.cpu().detach().numpy()
        if video_id not in video_clips:
            video_clips[video_id] = (att_clip, raw_vid_clip, title)
        else:
            extended_attention = np.append(video_clips[video_id][0], att_clip, axis=0)
            extended_video = np.append(video_clips[video_id][1], raw_vid_clip, axis=0)
            extended_title = np.append(video_clips[video_id][2], title)
            video_clips[video_id] = (extended_attention, extended_video, extended_title)
        if args.save_frames:
            out_dir = os.path.join(output_dir, str(video_id))
            os.makedirs(out_dir, exist_ok=True)
            print('saving frames to out_dir', out_dir)
            if args.show or args.save_frames:
                for i in range(len(video)):  # For each frame in current video-clip
                    img = video[i]
                    att_img = att_video_clip[i]
                    title = f'{sample_name}-{i}-{corr}-{label}.jpg'
                    if args.show:
                        plt.imshow(img.squeeze().cpu(), cmap='Greys_r')
                        plt.imshow(att_img.squeeze().cpu(), cmap='jet', alpha=0.5)
                        plt.title(title)
                        plt.show()
                    if args.save_frames:
                        x = overlay(img, att_img)
                        cv2.imwrite(os.path.join(out_dir, title), x)
        if args.zip:
            os.system(f'zip -r {base_res_dir}.zip {base_res_dir}')
    if args.save_video:
        for video_id in video_clips:
            x = [overlay(vid, np.expand_dims(att_vid, axis=0)) for (att_vid, vid) in zip(video_clips[video_id][0],
                                                                                         video_clips[video_id][1])]
            frame_titles = video_clips[video_id][2]  # get titles for frames in video, to extract no. corrs & label
            frame_corrs = np.asarray([frame_title.split('-')[1] for frame_title in frame_titles])
            ratio_corr = (frame_corrs == 'CORR').sum() / len(frame_titles)
            print('ratio corr', ratio_corr)
            if ratio_corr > 0.91 or ratio_corr < 0.3:
                true_label = frame_titles[0].split('-')[-1][:-4]
                video_title = f'{video_id}-{ratio_corr:.2f}-{true_label}.jpg'
                vs = VideoSaver(video_title, x, out_dir=args.out_dir + '_video')
                vs.save_video()


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
    # Instantiate model with grad cam & evaluate
    model = medcam.inject(model, return_attention=True, backend='gcam')
    model.eval()
    get_save_grad_cam_images(val_data_loader, model, device, subset='valid')
    if args.train_set:
        train_data_loader = get_data_loader(train=True)
        get_save_grad_cam_images(train_data_loader, model, device, subset='train')


if __name__ == '__main__':
    args = parser.parse_args()
    main()