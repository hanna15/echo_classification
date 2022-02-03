import torch
from echo_ph.visual.video_saver import VideoSaver
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from utils.transforms import get_transforms
from echo_ph.data import EchoDataset
from echo_ph.models.resnet_3d import Res3DSaliency
import numpy as np
from torch.utils.data import DataLoader
import os
from utils.helpers import get_index_file_path
from scipy.ndimage import zoom
from scripts.visualisations.vis_grad_cam_temp import overlay


parser = ArgumentParser(
    description='Arguments for visualising grad cam',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_path', default=None, help='set to path of a model state dict, to evaluate on. '
                                                       'If None, use resnet18 pretrained on Imagenet only.')
parser.add_argument('--out_dir', default='vis_3d_saliency', help='Name of directory storing the results')
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
parser.add_argument('--save_frames', action='store_true', help='If to save grad cam images on a frame-level')
parser.add_argument('--save_video', action='store_true', help='If also to save video visualisations')
parser.add_argument('--show', action='store_true', help='If to show grad cam images')
parser.add_argument('--train_set', action='store_true', help='Also get grad cam for the images in the training set, '
                                                             'with random augmentation (type 3)')
parser.add_argument('--segm_only', action='store_true', help='Only evaluate on the segmentation masks')
parser.add_argument('--video_ids', default=None, nargs='+', type=int, help='Instead of getting results acc.to index file, '
                                                                 'get results for specific video ids')
parser.add_argument('--dynamic', action='store_true', help='If run on dynamic dataset')


def get_data_loader(train=False):
    if args.dynamic:
        index_file_path = os.path.join('index_files', 'dynamic_test_index.npy')
    elif args.video_ids is None:
        index_file_path = get_index_file_path(args.k, args.fold, args.label_type, train=train)
    else:
        index_file_path = None
    if args.dynamic:
        label_path = os.path.join('label_files', 'dynamic_test_labels.pkl')
    else:
        label_path = os.path.join('label_files', 'labels_' + args.label_type + '.pkl')
    aug_type = 5 if train else 0
    dyn = False
    transforms = get_transforms(index_file_path, dataset_orig_img_scale=args.scale, resize=224,
                                augment=aug_type, fold=args.fold, valid=(not train), view=args.view,
                                crop_to_corner=args.crop, segm_mask_only=args.segm_only)
    dataset = EchoDataset(index_file_path, label_path, cache_dir=args.cache_dir,
                          transform=transforms, scaling_factor=args.scale, procs=args.n_workers,
                          percentile=args.max_p, view=args.view, num_rand_frames=1,
                          segm_masks=args.segm_only, dynamic=dyn, temporal=True, clip_len=12)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.n_workers)
    return data_loader


def main():
    out_dir = args.out_dir + '_fold_' + str(args.fold)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = 2 if args.label_type.startswith('2') else 3
    model = Res3DSaliency(num_classes=num_classes, model_type='r3d_18')
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    model = model.to(device)
    data_loader = get_data_loader()

    for batch in data_loader:
        inp = batch['frame'][args.view].to(device).transpose(2, 1)  # Reshape to: (batch_size, channels, seq-len, W, H)
        sample_name = batch['sample_name'][0]
        label = batch['label'][0].item()
        out, layerout = model(inp)
        pred = torch.max(out, dim=1).indices[0].item()
        corr = 'CORR' if label == pred else 'WRONG'

        # Get Grad-CAM: Fetch weights of fc model, fetch last layer activations, multiply! Resize, norm,
        pred_weights = model.fc.weight.data.detach().cpu().numpy().transpose()
        layerout = layerout.detach().cpu()[0].numpy()  # remove batch part
        layerout = layerout.swapaxes(0, 3)  # I added this
        cam = np.zeros(dtype=np.float32, shape=layerout.shape[0:3])
        for i, w in enumerate(pred_weights[:, label]):
            # Compute cam for every kernel
            cam += w * layerout[:, :, :, i]

        # Resize CAM to frame level
        cam = cam.swapaxes(1, 0)  # I added
        # cam = zoom(cam, (2, 32, 32))  # output map is 8x7x7, so multiply to get to 16x224x224 (original image size)
        # TODO: I need to replace this for my own size
        cam = zoom(cam, (6, 16, 16))  # out cam is 2x14x14, so multiply by 6x16x16 to get 12x224z224 (out inp size)

        # Normalize
        cam -= np.min(cam)
        cam /= np.max(cam) - np.min(cam)

        # make dirs and filenames
        # example_name = os.path.basename(args.frame_dir)
        # heatmap_dir = os.path.join(args.base_output_dir, example_name, str(args.label), "heatmap")
        # focusmap_dir = os.path.join(args.base_output_dir, example_name, str(args.label), "focusmap")
        # for d in [heatmap_dir, focusmap_dir]:
        #     if not os.path.exists(d):
        #         os.makedirs(d)
        #
        # file = open(os.path.join(args.base_output_dir, example_name, str(args.label), "info.txt"), "a")
        # file.write("Visualizing for class {}\n".format(args.label))
        # file.write("Predicted class {}\n".format(pred))
        # file.close()

        # produce heatmap and focusmap for every frame and activation map
        inp = inp.detach().cpu().numpy()[0]  # first one in the batch
        inp = inp.swapaxes(0, 1)  # no_frames, ch, w, h
        heat_video = []
        for i in range(0, cam.shape[0]):
            curr_cam = cam[i]
            curr_img = inp[i]
            heatframe = overlay(curr_img, curr_cam)
            heat_video.append(heatframe)
        print(len(heat_video))

        #for i in range(0, cam.shape[0]):
        #    #   Create colourmap
        #    heatmap = cv2.applyColorMap(np.uint8(255 * cam[i]), cv2.COLORMAP_JET)
        #    # Create frame with heatmap
        #    heatframe = (heatmap // 2) + (np.asarray(inp[i]).astype(np.uint8).transpose(1, 2, 0) // 2)
        #    heat_video.append(heatframe)

        title = f'{sample_name}-{corr}-{label}.jpg'
        vs = VideoSaver(title, heat_video, out_dir=out_dir)
        vs.save_video()


if __name__ == '__main__':
    args = parser.parse_args()
    main()
