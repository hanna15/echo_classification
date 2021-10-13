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


def get_num_max_exp_frames(label):
    """
    Given the provided label / class, return how many max expansion frames should be extracted for the video
    :param label: The label of the video
    :return: The number of max expansion frames to be extracted.
    """
    # TODO: Calculate it based on i) science, ii) label distribution (from all labels)
    # Now it is just a hax to provide somewhat more even class distribution
    if label == 0:  # normal echo
        return 4
    if label == 1:  # little PH
        return 12
    return 12  # label == 2 is medium to high PH


class EchoDataset(Dataset):
    def __init__(self, file_list_path, label_file_path, videos_dir=None, cache_dir=None,
                 transform=None, scaling_factor=0.5, procs=3):
        """
        Dataset for echocardiogram processing and classification in PyTorch.
        :param file_list_path: Path to a numpy file, listing all sample names to use in this dataset.
        :param label_file_path: Path to pickle file, containing a dictionary of labels per sample name.
        :param videos_dir: Path to folder holding raw videos, if raw videos should be loaded. Else, None.
        :param cache_dir: Path to the folder holding the processed, cached videos, if those should be used. Else, None.
        :param transform: Torchvision transpose to apply to each sample in this dataset. If no transform, set to None.
        :param scaling_factor: What scaling factor cached videos have. If using raw videos, scaling factor is not used.
        :param procs: How many processes to use for processing this dataset.
        """

        self.frames = []
        self.targets = []
        self.transform = transform

        # Paths
        self.videos_dir = videos_dir
        if cache_dir is None:
            self.cache_dir = None
        else:
            self.cache_dir = os.path.join(os.path.expanduser(cache_dir), str(scaling_factor))
        self.label_path = label_file_path

        samples = np.load(file_list_path)
        t = time()
        with mp.Pool(processes=procs) as pool:
            for frames, label in pool.map(self.load_sample, samples):
                if frames is not None and label is not None:
                    for frame in frames:  # each frame becomes an individual sample (with the same label)
                        self.frames.append(frame)
                        self.targets.append(label)
        t = time() - t
        self.num_samples = len(self.frames)
        self.labels, cnts = np.unique(self.targets, return_counts=True)
        # Calculate class weights for weighted loss
        self.class_weights = class_weight.compute_class_weight('balanced', self.labels, self.targets)
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
            curr_video_path = os.path.join(self.videos_dir, str(sample) + 'KAPAP.mp4')  # TODO: Generalise
        else:  # Use cached videos
            curr_video_path = os.path.join(self.cache_dir, str(sample) + 'KAPAP.npy') # TODO: Generalise
        if not os.path.exists(curr_video_path):
            print(f'Skipping sample {sample}, as the video path {curr_video_path} does not exist')
            return None, None
        try:  # Try to get segmentations
            sample_w_ending = str(sample) + 'KAPAP'
            segm = SegmentationAnalyser(sample_w_ending, 'segmented_results')
        except:
            print(f'Skipping sample {sample}, as the segmented results for it does not exist')
            return None, None

        # === Get labels ===
        with open(self.label_path, 'rb') as label_file:
            all_labels = pickle.load(label_file)
        label = all_labels[sample]

        # === Get max expansion frames ===
        num_max_frames = get_num_max_exp_frames(label)
        if self.cache_dir is None:  # load raw video and process
            segmented_video = load_and_process_video(curr_video_path)
        else:  # load already processed numpy video
            segmented_video = np.load(curr_video_path)
        max_exp_frame_nrs = segm.extraxt_max_frames(num_max_frames) # get frame-ids for  max expansion frames
        max_exp_frames = segmented_video[max_exp_frame_nrs]
        return max_exp_frames, label

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.targets[idx]
        frame = self.transform(frame)
        sample = {'label': label, 'frames': frame}
        return sample
