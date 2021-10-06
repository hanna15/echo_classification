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
from models.unet import Unet

"""
This script segments echo videos using pre-trained models from the echocv repo: bitbucket.org/rahuldeo/echocv
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
parser.add_argument('--max_frames', default=3, help='Max number of frames to do segmentation prediction on')
parser.add_argument('--samples', default=None, nargs='+',
                    help='f.'
                         'Input space names of the desired samples /file names (including ending) after this flag')
parser.add_argument('--save_visualisations', action='store_true', help='set this flag to save the visuals of '
                                                                       'segmentation mask for each image / frame ')

# /Users/hragnarsd/Documents/masters/echo_segmentation/echo_preprocessing/echocv/models'


def save_img(index, orig_image, segm_frame, outpath, videofile):
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


def save_frames(segm_frames, view, outpath, videofile):

    with bz2.BZ2File(os.path.join(outpath, videofile + "-frames.segment_pbz2"), 'wb') as file:
        pickle.dump(segm_frames, file)

    with open(os.path.join(outpath, videofile + "-segmentation_label.json"), 'w') as file:
        json.dump(segmentation_labels[view], file)


model_dict = {
            'psax': {
                'label_dim': 4,
                'restore_path': 'psax_45_20_all_model.ckpt-9300'
            },
            'plax': {
                'label_dim': 7,
                'restore_path': 'plax_45_20_all_model.ckpt-9600'
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
            'psax':{
                'lv': 2,  # left ventricle
                'rv': 3,  # right ventricle
                'lv-o': 1  # left ventricle outer tissue
            }
}


def main():
    args = parser.parse_args()
    model_views = ['psax', 'plax']  # The views of interest.

    # NN Parameters
    mean = 24
    weight_decay = 1e-12
    learning_rate = 1e-4
    maxout = False
    # demo: 30, 55, 97
    if args.samples is None:
        video_files = os.listdir(args.videos_path)
    else:
        video_files = args.samples
    for video_file in video_files:
        if video_file.endswith('.npy') or video_file.endswith('.mp4'):
            video_name = video_file[:-4]
            out_dir = os.path.join(args.out_root_dir, video_name)
            for view in model_views:
                print(f'processing video {video_name} for view {view}')
                view_out_dir = os.path.join(out_dir, view)
                os.makedirs(view_out_dir, exist_ok=True)
                print('Results will be saved to', view_out_dir)
                g_4 = tf.Graph()
                with g_4.as_default():
                    label_dim = model_dict[view]['label_dim']
                    checkpoint_path = model_dict[view]['restore_path']
                    sess = tf.compat.v1.Session()
                    model = Unet(mean, weight_decay, learning_rate, label_dim, maxout=maxout)
                    sess.run(tf.compat.v1.local_variables_initializer())
                with g_4.as_default():
                    saver = tf.compat.v1.train.Saver()
                    saver.restore(sess, os.path.join(args.models_path, checkpoint_path))
                if video_file.endswith('.npy'):
                    video_frames = np.load(os.path.join(args.videos_path, video_file))
                else:
                    print("Not yet implemented => ToDO: Add code snippet to take raw videos")
                    video_frames = None
                num_frames = len(video_frames)
                num_frames = min(num_frames, args.max_frames)
                frames_resized = []
                transform = transforms.Compose([transforms.ToPILImage(),
                                                transforms.Grayscale(),
                                                transforms.Resize(size=(384, 384), interpolation=InterpolationMode.BICUBIC)
                                                ])
                for i in range(num_frames):
                    img_frame = video_frames[i]
                    transformed_frame = np.array(transform(img_frame))
                    frames_resized.append(transformed_frame)
                frames_to_predict = np.array(frames_resized, dtype=np.float64).reshape((len(frames_resized), 384, 384, 1))
                predicted_frames = []
                if args.save_visualisations:
                    for i in range(num_frames):
                        predicted_frames.append(np.argmax(model.predict(sess, frames_to_predict[i:i + 1])[0, :, :, :], 2))
                        save_img(i, frames_resized[i], predicted_frames[i], view_out_dir, video_name)
                save_frames(predicted_frames, view, view_out_dir, video_name)


if __name__ == '__main__':
    main()
