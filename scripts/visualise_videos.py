import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from echo_ph.data.echo_dataset import load_and_process_video
import matplotlib.pyplot as plt
"""
Script to visualise frames of echo videos
"""


parser = ArgumentParser(
    description='Segment echo videos',
    formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--video_path', default='/Users/hragnarsd/.heart_echo/0.5_backup/96KAPAP.npy',
                    help='Path to the video to visualise')
parser.add_argument('--sampling_period', type=int, default=2,
                    help='Input integer i, to visualise every i-th frame')


def main():
    args = parser.parse_args()
    if args.video_path.endswith('.npy'):
        video = np.load(args.video_path)
    else:
        video = load_and_process_video(args.video_path)

    for i in range(0, len(video), args.sampling_period):
        plt.imshow(video[i])
        plt.title('frame nr' + str(i))
        plt.show()


if __name__ == '__main__':
    main()
