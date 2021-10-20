import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from echo_ph.data.echo_dataset import load_and_process_video
import matplotlib.pyplot as plt

from torchvision import transforms
from utils import transforms as util_trans
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
parser.add_argument('--vis_transforms', action='store_true', help='Input integer i, '
                                                                  'to visualise every i-th frame')


def show_frame(frame, title: str):
    if frame.shape[0] == 1:  # if grayscale, i.e. only 1 colour-channel
        frame = frame.squeeze(0)
    plt.imshow(frame, cmap='Greys_r')
    plt.title(title)
    plt.show()


def main():
    args = parser.parse_args()
    if args.video_path.endswith('.npy'):
        video = np.load(args.video_path)
    else:
        video = load_and_process_video(args.video_path)
    if args.vis_transforms:
        # Show frame nr 5 before and after each major transform
        frame = video[5]
        show_frame(frame, 'Before')
        base_t = transforms.Compose([util_trans.HistEq(),
                                     transforms.ToPILImage(),
                                     transforms.Resize(size=(128, 128), interpolation=transforms.InterpolationMode.BICUBIC),
                                     transforms.ToTensor()]) #,
                                     # util_trans.Normalize()])
        frame = base_t(frame)
        show_frame(frame, 'After BaseTrans')
        trans_names = ['Sharpen', 'BrightAdjust', 'Gamma', 'ResizeCrop', 'ResizePad', 'Rotate', 'Translate']
        sharp = util_trans.RandomSharpness()
        bright = util_trans.RandomBrightnessAdjustment()
        gamma = util_trans.RandomGammaCorrection()
        resize_crop = util_trans.RandResizeCrop(1.15)
        resize_pad = util_trans.RandResizePad(0.85, False)
        rotate = util_trans.Rotate()
        translate = util_trans.Translate()
        adv_t = [sharp, bright, gamma, resize_crop, resize_pad, rotate, translate]
        adv_t_composed = transforms.Compose([sharp, bright, gamma, resize_pad, rotate, translate])
        # for i in range(4):
        for trans_name, trans in zip(trans_names, adv_t):
            #if trans_name == 'BrightAdjust':
            # t = transforms.Compose([trans])
            t = transforms.Compose([util_trans.RandomNoise()])
            tmp_frame = t(frame)
            show_frame(tmp_frame, trans_name)

        # for i in range(3):
        #     tmp_frame = adv_t_composed(frame)
        #     show_frame(tmp_frame, 'ALL_' + str(i))

    else:  # print each frame in video with given sampling period
        for i in range(0, len(video), args.sampling_period):
            plt.imshow(video[i])
            plt.title('frame nr' + str(i))
            plt.show()


if __name__ == '__main__':
    main()
