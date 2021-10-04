from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import multiprocessing as mp
from time import time
from sklearn.utils import class_weight
from heart_echo.Processing import ImageUtilities, VideoUtilities
from heart_echo.Helpers import Helpers


def load_and_process_video(video_path):
    cropped_frames, segmented_points = Helpers.load_video(video_path)
    m, b = VideoUtilities.calculate_line_parameters(*segmented_points)
    cropped_frames = [ImageUtilities.fill_side_of_line(frame, m, b) for frame in cropped_frames]
    segmented_video = VideoUtilities.segment_echo_video(cropped_frames, *segmented_points)
    return segmented_video


class EchoDataset(Dataset):
    def __init__(self, videos_dir='/Users/hragnarsd/Documents/masters/videos/Heart_Echo', cache_dir='~/.heart_echo',
                 transform=None, procs=3, file_list_path='train_samples.npy', label_file_path='labels3.pkl',
                 scaling_factor=0.5):

        self.frames = []
        self.labels = []
        self.transform = transform

        # Paths
        self.videos_dir = videos_dir
        self.cache_dir = os.path.join(os.path.expanduser(cache_dir), str(scaling_factor))
        self.label_path = label_file_path

        samples = np.load(file_list_path)
        t = time()
        with mp.Pool(processes=procs) as pool:
            for frame, label in pool.map(self.load_sample, samples):
                if frame is not None and label is not None:
                    self.frames.append(frame)
                    self.labels.append(label)
            # ToDo the body -> Adding each sample to an array
        t = time() - t
        self.num_samples = len(self.frames)
        # Calculate class weights for weighted loss
        self.class_weights = class_weight.compute_class_weight('balanced', np.unique(self.labels), self.labels)
        print(f'Loaded Dataset with {self.num_samples} samples in {t:.2f} seconds')

    def load_sample(self, sample):
        """
        Load line regions and program for a given sample
        :param sample: Sample from the file list paths.
        :return: (line regions, parsed program, sample name)
        """
        print('loading sample', sample)
        curr_video_cache_path = os.path.join(self.cache_dir, str(sample) + 'KAPAP.npy')
        # curr_video_path = os.path.join(self.videos_dir, str(sample) + 'KAPAP.mp4')  # TODO: Generalise
        try:
            # segmented_video = load_and_process_video(curr_video_path) # If not cache
            segmented_video = np.load(curr_video_cache_path)
        except:
            print(f"error - video {curr_video_cache_path} not found")
            return None, None
        # TODO: Change this into extracting the relevant frame(s) from a video
        frame_nr_1 = segmented_video[1]

        # Get labels
        with open(self.label_path, 'rb') as label_file:
            all_labels = pickle.load(label_file)
        label = all_labels[sample]
        return frame_nr_1, label

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        frame = self.transform(frame)
        sample = {'label': label, 'frame': frame}
        return sample