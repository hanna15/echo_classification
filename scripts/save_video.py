# Python program to save a video using OpenCV

import cv2
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from echo_ph.visual.video_saver import VideoSaver

parser = ArgumentParser(
    description='Arguments for generating video',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--in_dir', default='grad_cam_vis', help='Path to directory storing grad cams for models')
parser.add_argument('--model', default='fold0_resnet_adamw_lt_2d.k10.lr_0.001.batch_64.wd_0.001.me_300_pre_aug4_bal_'
                                       'histrand_n10_e24', help='name of the model, for which this grad cam is')
parser.add_argument('--subset', default='valid')
parser.add_argument('--video_ids', nargs='+', help='list of video ids to create video for')
parser.add_argument('--in_path', default=None,
                    help='Full path to directory storing all frames of one video, if not many videos in a directory.')
parser.add_argument('--out_dir', default='vis_videos',
                    help='Path to directory storing all frames of one video')
parser.add_argument('--fps', type=int, default=10, help='sets the video speed, i.e. fps')
parser.add_argument('--max_frames', type=int, default=50, help='Set the max no. frames to render to a video')


def main():
    if args.in_path is not None:
        video_dirs = [args.in_path]
    else:
        video_dirs = [os.path.join(args.in_dir, args.model, args.subset, video_id) for video_id in args.video_ids]
    for video_dir, video_id in zip(video_dirs, args.video_ids):
        frames = []
        for frame_file in os.listdir(video_dir):
            frame = cv2.imread(os.path.join(video_dir, frame_file))
            frames.append(frame)
        vs = VideoSaver(video_id, frames, out_dir=args.out_dir, max_frames=args.max_frames, fps=args.fps)
        vs.save_video()
        # size = (frames[0].shape[0], frames[0].shape[1])
        # out_path = os.path.join(args.out_dir, video_id + '.avi')
        # result = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'MJPG'), args.fps, size)
        #
        # frames = frames[0:args.max_frames]  # Only first 100 frames, or approx first 5 heart-beats
        # print(len(frames))
        # for frame in frames:
        #     # Write the frame into the file 'filename.avi'
        #     result.write(frame)
        # result.release()
        # # Closes all the frames
        # cv2.destroyAllWindows()
        print("The video was successfully saved")


if __name__ == '__main__':
    args = parser.parse_args()
    main()