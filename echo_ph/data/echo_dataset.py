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
import cv2


def load_dynamic_video(filename: str) -> np.ndarray:
    """Loads a video from a file.

    Args:
        filename (str): filename of video

    Returns:
        A np.ndarray with dimensions (channels=3, frames, height, width). The
        values will be uint8's ranging from 0 to 255.

    Raises:
        FileNotFoundError: Could not find `filename`
        ValueError: An error occurred while reading the video
    """

    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)

    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError("Failed to load frame #{} of {}.".format(count, filename))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count, :, :] = frame

    # v = v.transpose((3, 0, 1, 2))
    v = VideoUtilities.convert_to_gray(v)
    return v.astype(np.uint8)


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
                 min_expansion=False, num_rand_frames=None, segm_masks=False, temporal=False, period=1, clip_len=0,
                 all_frames=False, max_frame=None, video_ids=None, dynamic=False, regression=False):
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
        :param view: What model view to use. If get data for multi-modal models (i.e. multi-view models),
                      provide a list of all views. This will ensure each model view has same frames numbers.
        :param min_expansion: If True, then pick minimum expansion frames instead of maximum expansion frames
        :param num_rand_frames: If None, get min/max expansion frames. Else, set to numner of random frames to pick per video
        """

        self.dynamic = dynamic
        if isinstance(view, list) and num_rand_frames is None and all_frames is None:
            print("Multiple views are only supported in conjunction with random frames or all frames,"
                  "not min/max frames - as those differ between views")
        self.views = view if isinstance(view, list) else [view]
        self.frames = dict.fromkeys(self.views, [])
        self.targets = []
        self.sample_names = []
        self.transform = transform
        self.videos_dir = videos_dir
        self.temporal = temporal
        self.period = period  # How much to sub-sample frames in a clip
        self.clip_len = clip_len  # How many frames in a clip => total frame span is period * clip_len
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
        self.all_frames = all_frames
        self.max_frame = max_frame
        self.segm_masks = segm_masks
        self.regression = regression
        self.view_to_segmodel_view = {  # When training on given view, what segmentation pretrained model view to use
            'KAPAP': 'psax',
            'CV': 'a4c'
        }
        if index_file_path is not None:
            samples = np.load(index_file_path)
        else:
            samples = video_ids
        t = time()
        with mp.Pool(processes=procs) as pool:
            for frames_per_view, label, sample_names in pool.map(self.load_sample, samples):
                if frames_per_view is not None and label is not None and sample_names is not None:
                    view_no = 0
                    for frames, view in zip(frames_per_view, self.views):
                        for frame, sample_name in zip(frames, sample_names):  # Each frame becomes an individual sample (with the same label)
                            self.frames[view].append(frame)
                            self.targets.append(label)
                            self.sample_names.append(sample_name)
                        view_no += 1
        t = time() - t
        self.num_samples = len(self.frames)
        self.labels, cnts = np.unique(self.targets, return_counts=True)
        # Calculate class weights for weighted loss
        if self.regression:
            self.example_weights = None
        else:
            self.class_weights = class_weight.compute_class_weight('balanced', classes=self.labels, y=self.targets)
            if len(self.class_weights) <= max(self.labels):  # we have a missing label = not calculate example weights (hax)
                self.example_weights = None
            else:
                self.example_weights = [self.class_weights[t] for t in self.targets]
        print(f'Loaded Dataset with {self.num_samples} samples in {t:.2f} seconds. Label distribution:')
        for label, cnt in zip(self.labels, cnts):  # Print number of occurrences of each label
            print(label, ':', cnt)

    def get_frame_nrs(self, total_len):
        if self.all_frames:
            if self.max_frame is None:
                max_frame = total_len
            else:
                max_frame = min(total_len, self.max_frame)  # Entire video, or up to max frame
            if self.temporal:
                # get starting points, s.t. start + (clip_len * sp) covers all video
                max_frame = max_frame - (self.clip_len * self.period)
                return np.asarray(range(0, max_frame, self.clip_len * self.period))
            # Else, if get all frames, but NOT temporal - just return sequence of corr len
            return np.asarray(range(0, max_frame))
        # Get random frames
        max_frame = total_len - (self.clip_len * self.period)
        frame_nrs = np.random.randint(0, max_frame, self.num_rand_frames)
        return frame_nrs

    def get_frames(self, frame_nrs, all_frames):
        """
        Get frames to train on, given frame numbers and all frames (i.e. entire video or entire segmentation masks)
        :param frame_nrs: The frame numbers
        :param all_frames: All frames of a given video or segmentation mask
        :return: The frames to be used for this sample, depending on if temporal or spatial approach
        """
        if self.temporal:
            start_frame_nrs = frame_nrs
            frame_sequences = []
            for s in start_frame_nrs:
                seq = np.asarray(range(s, s + self.clip_len * self.period, self.period))
                frame_sequences.append(seq)
            frames = all_frames[np.asarray(frame_sequences)]
        else:
            frames = all_frames[frame_nrs]
        return frames

    def load_sample(self, sample):
        """
        Load line regions and program for a given sample
        :param sample: Sample from the file list paths.
        :return: (line regions, parsed program, sample name)
        """
        #if np.random.random() < 0.7:
        #    return None, None, None
        views = self.views
        videos = []
        for view in views:
            if self.dynamic:
                curr_video_path = os.path.join('/Users/hragnarsd/Documents/masters/dynamic/a4c-video-dir/Videos',
                                               sample + '.avi')  # TODO: Don't have it hard-coed
            elif self.cache_dir is None:  # Use raw videos, as no cached processed videos provided
                curr_video_path = os.path.join(self.videos_dir, str(sample) + view + '.mp4')  # TODO: Generalise
            else:  # Use cached videos
                curr_video_path = os.path.join(self.cache_dir, str(sample) + view + '.npy')  # TODO: Generalise
            if not os.path.exists(curr_video_path):
                print(f'Skipping sample {sample}, as the video path {curr_video_path} does not exist')
                return None, None, None
            if self.segm_masks:  # Train only on segmentation mask frames
                sample_w_ending = str(sample) + view
                segm = SegmentationAnalyser(sample_w_ending,
                                            os.path.join('segmented_results', str(self.scaling_factor)),
                                            model_view=self.view_to_segmodel_view[view])
                segm_video = segm.get_segm_mask()
            else:
                if self.dynamic:
                    segm_video = load_dynamic_video(curr_video_path).astype(np.float32)
                elif self.cache_dir is None:  # load raw video and process
                    segm_video = load_and_process_video(curr_video_path)
                else:  # load already processed numpy video
                    segm_video = np.load(curr_video_path)
            videos.append(segm_video)

        # === Get frames for video ===
        if self.num_rand_frames or self.all_frames:
            total_len = min([len(vid) for vid in videos])  # the len will be the len of the shorter video
            frame_nrs = self.get_frame_nrs(total_len=total_len)
        else:  # Get max or min expansion frames, acc. to segmentation percentile
            # This can only be reached in case a single model (bc. otherwise error in construction)
            # Thus, can just pick view[0] - will be the only one!
            sample_w_ending = str(sample) + views[0]
            # Todo: Generalise segm result dir
            segm = SegmentationAnalyser(sample_w_ending, os.path.join('segmented_results',
                                                                      str(self.scaling_factor)),
                                        model_view=self.view_to_segmodel_view[views[0]])
            frame_nrs = segm.extract_max_percentile_frames(percentile=self.max_percentile,
                                                           min_exp=self.min_expansion)
        sample_names = [str(sample) + '_' + str(frame_nr) for frame_nr in frame_nrs]

        frames_per_view = []
        for segm_video in videos:
            frames = self.get_frames(frame_nrs, segm_video)
            frames_per_view.append(frames)
        # === Get labels ===
        with open(self.label_path, 'rb') as label_file:
            all_labels = pickle.load(label_file)
        if sample not in all_labels:  # ATH! When proper index files used, this should not happen
            return None, None, None
        label = all_labels[sample]
        if self.regression:  # normalise labels
            max_label = max(all_labels.values())
            label = label/max_label
        return frames_per_view, label, sample_names

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = self.targets[idx]
        sample_name = self.sample_names[idx]
        frame_per_view = dict.fromkeys(self.views, [])
        for view in self.views:
            frames = self.frames[view][idx].astype(np.uint8)
            if self.temporal:
                frames = list(frames)
            s = (frames, sample_name.split('_')[0] + view)
            frames = self.transform(s)
            frame_per_view[view] = frames  # over-write with transformed frames
            if self.visualise_frames:
                if self.temporal:
                    for i, f in enumerate(frames):
                        plt.imshow(f.squeeze(0), cmap='Greys_r')
                        plt.title(str(label) + ' - ' + str(sample_name) + '-' + str(i))
                        plt.show()
                else:
                    plt.imshow(frames.squeeze(0), cmap='Greys_r')
                    plt.title(str(label) + ' - ' + str(sample_name))
                    plt.show()
        sample = {'label': label, 'frame': frame_per_view, 'sample_name': sample_name}
        return sample
