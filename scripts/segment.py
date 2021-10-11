import tensorflow as tf
import os
import numpy as np
import json
import pickle
import bz2
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from echo_ph.models import Unet

"""
This script segments echo videos using pre-trained models from the echo-cv repo: bitbucket.org/rahuldeo/echocv
"""


parser = ArgumentParser(
    description='Segment echo videos',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--models_path', default='/Users/hragnarsd/Documents/masters/segmentation_models',
                    help='Path to the directory storing the pre-trained models (checkpoints)')
parser.add_argument('--videos_path', default='/Users/hragnarsd/.heart_echo/0.5_backup',
                    help='Path to the directory storing the echo videos to be segmented.')
parser.add_argument('--out_root_dir', default='segmented_results',
                    help='Path to the directory that should store the resulting segmentation maps')
parser.add_argument('--max_frames', type=int, default=1000, help='Max number of frames to do segmentation prediction on')
parser.add_argument('--sampling_period', type=int, default=1,
                    help='If sample each frame, set to 1 (default). To sample every x-th frame, set to x.')
parser.add_argument('--samples', default=None, nargs='+',
                    help='Set this flag to segment only specific samples (videos) instead of all. Input space seperated'
                         'names of the desired samples/file names (including ending) after this flag')
parser.add_argument('--model_views', default='psax', choices=['psax', 'plax', 'all', 'a4c'],
                    help='What echocv-views to use for segmentation. Currently only psax and plax supported.')
parser.add_argument('--our_view', default='KAPAP', choices=['KAPAP', 'KAAP', 'CV'],
                    help='What view to use for our data. Currently only KAPAP, KAAP and CV supported.')
parser.add_argument('--save_visualisations', action='store_true', help='set this flag to save the visuals of '
                                                                       'segmentation mask for each image / frame ')


def save_segmentation_visuals(index, orig_image, segm_frame, outpath, videofile):
    """
    Saves each frame / image of an echo video, as well as an overlay of it's corresponding segmentation mask
    :param index: Index of this frame (e.g. if 4th frame in a video, index=4)
    :param orig_image: The original frame / image
    :param segm_frame: The predicted segmentation map for the frame
    :param outpath: The path to the directory storing the results
    :param videofile: Name of the video / sample name.
    :return:
    """
    segm_frame = segm_frame.astype(np.float32)
    segm_frame[segm_frame == 0] = np.nan
    os.makedirs(outpath, exist_ok=True)

    plt.figure(figsize=(7, 7))
    plt.axis('off')
    plt.imshow(segm_frame, cmap='Set3')
    plt.savefig(outpath + '/' + videofile + '_' + str(index) + '_' + 'seg.png')
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.axis('off')
    plt.imshow(orig_image, cmap='Greys_r')
    plt.savefig(outpath + '/' + videofile + '_' + str(index) + '_' + 'orig.png')
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.axis('off')
    plt.imshow(orig_image, cmap='Greys_r')
    plt.imshow(segm_frame, alpha=0.3)
    plt.savefig(outpath + '/' + videofile + '_' + str(index) + '_' + 'overlay.png')
    plt.close()
    print('')


def save_segm_map(segm_frames, view, outpath, videofile):
    """
    Saves the predicted segmentation maps for each frame in a video as a compressed pickle file.
    :param segm_frames: The predictions (segmented frames), with shape [num_frames, w, h, 1]
    :param view: The ehco-cv view used for this segmentation.
    :param outpath: Path to store the results.
    :param videofile: Name of the video-file / sample.
    """

    with bz2.BZ2File(os.path.join(outpath, videofile + "-frames.segment_pbz2"), 'wb') as file:
        pickle.dump(segm_frames, file)

    with open(os.path.join(outpath, videofile + "-segmentation_label.json"), 'w') as file:
        json.dump(segmentation_labels[view], file)


#TODO: Move dictionaries to a different file - maybe joint label file for segment and PH.

model_dict = {
            'psax': {
                'label_dim': 4,
                'restore_path': 'psax_45_20_all_model.ckpt-9300'
            },
            'plax': {
                'label_dim': 7,
                'restore_path': 'plax_45_20_all_model.ckpt-9600'
            },
            'a4c': {
                'label_dim': 6,
                'restore_path': 'a4c_45_20_all_model.ckpt-9000'
            }
}

segmentation_labels = {
            'plax': {
                'lv': 1,  # left ventricle
                'ivs': 2,  # interventricular septum
                'la': 3,  # left atrium
                'rv': 4,  # right ventricle
                'pm': 5,  # papilliary muscle
                'aor': 6  # aortic root (ascending)
            },
            'psax': {
                'lv': 2,  # left ventricle
                'rv': 3,  # right ventricle
                'lv-o': 1  # left ventricle outer tissue
            },
            'a4c': {
                'lv': 2, # left ventricle
                'rv': 3, # right ventricle
                'la': 4, # left atrium
                'ra': 5, # right atrium
                'lv-o': 1 # left ventricle outer tissue
            }
}


def main():
    args = parser.parse_args()
    if args.model_views == 'all':
        model_views = ['psax', 'plax']
    else:
        model_views = [args.model_views]

    # NN Parameters
    mean = 24
    weight_decay = 1e-12
    learning_rate = 1e-4
    maxout = False
    frame_size = 384

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Grayscale(),
                                    transforms.Resize(size=(frame_size, frame_size),
                                                      interpolation=InterpolationMode.BICUBIC)])

    # demo: 30, 55, 97
    if args.samples is None:
        video_files = os.listdir(args.videos_path)
    else:
        video_files = args.samples
    video_ending = args.our_view + '.npy'
    video_files = [str(vid) + video_ending for vid in video_files]
    for video_file in video_files:
        if video_file.endswith('.npy') or video_file.endswith('.mp4'):
            video_name = video_file[:-4]
            out_dir = os.path.join(args.out_root_dir, video_name)
            for view in model_views:
                print(f'processing video {video_name} for view {view}')
                view_out_dir = os.path.join(out_dir, view)
                os.makedirs(view_out_dir, exist_ok=True)
                print('Results will be saved to', view_out_dir)

                # === Get model ===
                graph = tf.Graph()
                with graph.as_default():
                    num_labels = model_dict[view]['label_dim']
                    checkpoint_path = model_dict[view]['restore_path']
                    sess = tf.compat.v1.Session()
                    model = Unet(mean, weight_decay, learning_rate, num_labels, maxout=maxout)
                    sess.run(tf.compat.v1.local_variables_initializer())
                with graph.as_default():
                    saver = tf.compat.v1.train.Saver()
                    saver.restore(sess, os.path.join(args.models_path, checkpoint_path))

                # === Get data ===
                if video_file.endswith('.npy'): # already processed video
                    video_frames = np.load(os.path.join(args.videos_path, video_file))
                else:
                    print("Not yet implemented => ToDO: Add code snippet to work on raw videos")
                    video_frames = None
                num_frames = len(video_frames)
                num_frames = min(num_frames, args.max_frames)  # If only process specified number of frames
                # Resizing each frame that we want to process
                frames_resized = []
                for i in range(0, num_frames * args.sampling_period, args.sampling_period):
                    img_frame = video_frames[i]
                    transformed_frame = np.array(transform(img_frame))
                    frames_resized.append(transformed_frame)
                print('Starting to predict')
                frames_to_predict = np.array(frames_resized, dtype=np.float64).reshape((len(frames_resized), frame_size, frame_size, 1))
                predicted_frames = np.argmax(model.predict(sess, frames_to_predict), -1)  # argmax max over last dim (labels)
                save_segm_map(predicted_frames, view, view_out_dir, video_name)
                if args.save_visualisations:
                    for i in range(num_frames):
                        save_segmentation_visuals(i, frames_resized[i], predicted_frames[i], view_out_dir, video_name)


if __name__ == '__main__':
    # Note: these samples would be good to have segmented (as these I have available now for training).
    # video_files = [60, 56, 93, 79, 38, 90, 87, 82, 31, 71, 36, 54, 40, 94, 72, 48, 62, 83, 59, 42, 70, 33, 49, 34, 81,
    #               92, 80, 104, 69, 88, 35, 53, 73, 99, 57, 41, 68, 91, 84, 61, 86, 63, 50, 78, 95, 74, 75, 103, 105,
    #               85, 98, 102, 58, 100, 37, 67, 89, 39]
    main()
