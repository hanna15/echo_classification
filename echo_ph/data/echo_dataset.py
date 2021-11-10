import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import multiprocessing as mp
from time import time
from sklearn.utils import class_weight
from heart_echo.Processing import ImageUtilities, VideoUtilities
from heart_echo.Helpers import Helpers
from echo_ph.data.segmentation import SegmentationAnalyser
import matplotlib.pyplot as plt


def load_and_process_video(video_path):
    """
    Load a single raw video and return the cropped and segmented video.
    :param video_path: Path to the raw video
    :return: Processed numpy video
    """
    cropped_frames, segmented_points = Helpers.load_video(video_path)
    m, b = VideoUtilities.calculate_line_parameters(*segmented_points)
    cropped_frames = [ImageUtilities.fill_side_of_line(frame, m, b) for frame in cropped_frames]
    segmented_video = VideoUtilities.segment_echo_video(cropped_frames, *segmented_points)
    return segmented_video


class EchoDataset(Dataset):
    def __init__(self, index_file_path, label_file_path, videos_dir=None, cache_dir=None,
                 transform=None, scaling_factor=0.5, procs=3, visualise_frames=False, percentile=90, view='KAPAP',
                 min_expansion=False, num_rand_frames=None, segm_masks=False):
        """
        Dataset for echocardiogram processing and classification in PyTorch.
        :param index_file_path: Path to a numpy file, listing all sample names to use in this dataset.
        :param label_file_path: Path to pickle file, containing a dictionary of labels per sample name.
        :param videos_dir: Path to folder holding raw videos, if raw videos should be loaded. Else, None.
        :param cache_dir: Path to the folder holding the processed, cached videos, if those should be used. Else, None.
        :param transform: Torchvision transpose to apply to each sample in this dataset. If no transform, set to None.
        :param scaling_factor: What scaling factor cached videos have. If using raw videos, scaling factor is not used.
        :param procs: How many processes to use for processing this dataset.
        :param visualise_frames: If visualise frames during training (after transformation)
        :param view: What model view
        :param min_expansion: If True, then pick minimum expansion frames instead of maximum expansion frames
        :param num_rand_frames: If None, get min/max expansion frames. Else, set to numner of random frames to pick per video
        """

        self.frames = []
        self.targets = []
        self.sample_names = []
        self.transform = transform
        self.videos_dir = videos_dir
        if cache_dir is None:
            self.cache_dir = None
        else:
            self.cache_dir = os.path.join(os.path.expanduser(cache_dir), str(scaling_factor))
        self.label_path = label_file_path
        self.visualise_frames = visualise_frames
        self.scaling_factor = scaling_factor
        self.max_percentile = percentile
        self.min_expansion = min_expansion
        self.num_rand_frames = num_rand_frames
        self.view = view
        self.segm_masks = segm_masks
        self.view_to_segmodel_view = {  # When training on given view, what segmentation pretrained model view to use
            'KAPAP': 'psax',
            'CV': 'a4c'
        }

        samples = np.load(index_file_path)
        t = time()
        with mp.Pool(processes=procs) as pool:
            for frames, label, sample_names in pool.map(self.load_sample, samples):
                if frames is not None and label is not None and sample_names is not None:
                    for frame, sample_name in zip(frames, sample_names):  # each frame becomes an individual sample (with the same label)
                        self.frames.append(frame)
                        self.targets.append(label)
                        self.sample_names.append(sample_name)
        t = time() - t
        self.num_samples = len(self.frames)
        self.labels, cnts = np.unique(self.targets, return_counts=True)
        # Calculate class weights for weighted loss
        self.class_weights = class_weight.compute_class_weight('balanced', classes=self.labels, y=self.targets)
        if len(self.class_weights) <= max(self.labels):  # we have a missing label = not calculate example weights (hax)
            self.example_weights = None
        else:
            self.example_weights = [self.class_weights[t] for t in self.targets]
        print(f'Loaded Dataset with {self.num_samples} samples in {t:.2f} seconds. Label distribution:')
        for label, cnt in zip(self.labels, cnts):  # Print number of occurrences of each label
            print(label, ':', cnt)

    def load_sample(self, sample):
        """
        Load line regions and program for a given sample
        :param sample: Sample from the file list paths.
        :return: (line regions, parsed program, sample name)
        """
        if self.cache_dir is None:  # Use raw videos, as no cached processed videos provided
            curr_video_path = os.path.join(self.videos_dir, str(sample) + self.view + '.mp4')  # TODO: Generalise
        else:  # Use cached videos
            curr_video_path = os.path.join(self.cache_dir, str(sample) + self.view + '.npy')  # TODO: Generalise
        if not os.path.exists(curr_video_path):
            print(f'Skipping sample {sample}, as the video path {curr_video_path} does not exist')
            return None, None, None

        # === Get labels ===
        with open(self.label_path, 'rb') as label_file:
            all_labels = pickle.load(label_file)
        label = all_labels[sample]

        if self.segm_masks:  # Train only on segmentation mask frames
            sample_w_ending = str(sample) + self.view
            # Todo: Generalise segm result dir
            segm = SegmentationAnalyser(sample_w_ending, os.path.join('segmented_results', str(self.scaling_factor)),
                                        model_view=self.view_to_segmodel_view[self.view])
            segm_mask = segm.get_segm_mask()
            if self.num_rand_frames:
                frame_nrs = np.random.randint(0, len(segm_mask), self.num_rand_frames)
            else:
                frame_nrs = segm.extract_max_percentile_frames(percentile=self.max_percentile,
                                                               min_exp=self.min_expansion)
            frames = segm_mask[frame_nrs]
        else:  # Train on video frames
            # === Get video ==
            if self.cache_dir is None:  # load raw video and process
                segmented_video = load_and_process_video(curr_video_path)
            else:  # load already processed numpy video
                segmented_video = np.load(curr_video_path)
            # === Get frames for video ===
            if self.num_rand_frames:
                # frame_nrs = np.random.randint(0, len(segmented_video), self.num_rand_frames)
                frame_nrs = np.random.randint(0, len(segmented_video) - 10, self.num_rand_frames)
            else:  # Get max or min expansion frames, acc. to segmentation percentile
                sample_w_ending = str(sample) + self.view
                # Todo: Generalise segm result dir
                segm = SegmentationAnalyser(sample_w_ending, os.path.join('segmented_results',
                                                                          str(self.scaling_factor)),
                                            model_view=self.view_to_segmodel_view[self.view])
                frame_nrs = segm.extract_max_percentile_frames(percentile=self.max_percentile,
                                                               min_exp=self.min_expansion)
            start_frame_nrs = frame_nrs
            frame_sequences = []
            for s in start_frame_nrs:
                seq = np.asarray(range(s, s + 10))  # tune 10-12, etc ?
                frame_sequences.append(seq)
            # frames = segmented_video[frame_nrs]
            frames = segmented_video[np.asarray(frame_sequences)]

        sample_names = [str(sample) + '_' + str(fram_nr) for fram_nr in frame_nrs]
        return frames, label, sample_names

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = self.targets[idx]
        sample_name = self.sample_names[idx]
        frame = self.frames[idx]
        #s = (frame, sample_name.split('_')[0] + self.view)
        #frame = self.transform(s)
        size = (len(frame), 1, 224, 224)
        trans_frames = torch.empty(size)
        for i, f in enumerate(frame):
            s = (f, sample_name.split('_')[0] + self.view)
            fr = self.transform(s)
            # trans_frames.append(fr)
            trans_frames[i] = fr
        frame = trans_frames
        if self.visualise_frames:
            plt.imshow(frame.squeeze(0), cmap='Greys_r')
            plt.title(str(label) + ' - ' + str(sample_name))
            plt.show()
        sample = {'label': label, 'frame': frame, 'sample_name': sample_name}
        return sample
